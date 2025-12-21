import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from config import ExperimentConfig, FLASH_ATTN_AVAILABLE
if FLASH_ATTN_AVAILABLE:
    from flash_attn import flash_attn_qkvpacked_func

# Standard Attention (Baseline)
class Head(nn.Module):
    """Single attention head - standard implementation"""

    def __init__(self, head_size: int, block_size: int, n_embd: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * (self.head_size**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention"""

    def __init__(self, num_heads, block_size, head_size, n_embd, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, block_size, n_embd, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FlashMultiHeadAttention(nn.Module):
    """Multi-head attention using FlashAttention"""

    def __init__(self, num_heads, block_size, head_size, n_embd, dropout):
        super().__init__()
        if not FLASH_ATTN_AVAILABLE:
            raise ImportError("flash-attn is required for FlashAttention")

        self.n_embd = n_embd
        self.num_heads = num_heads
        self.head_size = head_size

        # Combined QKV projection
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_size)

        # FlashAttention expects (B, T, 3, H, D)
        # Apply causal mask
        out = flash_attn_qkvpacked_func(
            qkv, dropout_p=self.dropout if self.training else 0.0, causal=True
        )

        # Reshape and project
        out = out.reshape(B, T, C)
        out = self.proj(out)
        return out


class LocalMultiHeadAttention(nn.Module):
    """Multi-head attention with local/sliding window"""

    def __init__(
        self, num_heads, block_size, head_size, n_embd, dropout, window_size=128
    ):
        super().__init__()
        if not FLASH_ATTN_AVAILABLE:
            raise ImportError("flash-attn is required for Local Attention")

        self.n_embd = n_embd
        self.num_heads = num_heads
        self.head_size = head_size
        self.window_size = window_size

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_size)

        # Use FlashAttention with window_size parameter
        out = flash_attn_qkvpacked_func(
            qkv,
            dropout_p=self.dropout if self.training else 0.0,
            causal=True,
            window_size=(self.window_size, 0),  # (left_window, right_window)
        )

        out = out.reshape(B, T, C)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    """MLP block"""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block with configurable attention"""

    def __init__(self, n_embd, block_size, n_head, dropout, config: ExperimentConfig):
        super().__init__()
        head_size = n_embd // n_head

        # Choose attention mechanism based on config
        if config.use_flash_attention:
            self.sa = FlashMultiHeadAttention(
                n_head, block_size, head_size, n_embd, dropout
            )
        elif config.use_local_attention:
            self.sa = LocalMultiHeadAttention(
                n_head,
                block_size,
                head_size,
                n_embd,
                dropout,
                window_size=config.local_attention_window,
            )
        else:
            self.sa = MultiHeadAttention(n_head, block_size, head_size, n_embd, dropout)

        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """GPT model with configurable optimizations"""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.block_size = config.block_size

        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)

        self.blocks = nn.Sequential(
            *[
                Block(
                    config.n_embd,
                    config.block_size,
                    config.n_head,
                    config.dropout,
                    config,
                )
                for _ in range(config.n_layer)
            ]
        )

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        self.apply(self._init_weights)

        # Enable gradient checkpointing if requested
        if config.use_gradient_checkpointing:
            self.gradient_checkpointing_enable()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency"""
        self._gradient_checkpointing = True

        # Wrap each block with checkpoint
        for i, block in enumerate(self.blocks):
            self.blocks[i] = self._wrap_block_with_checkpoint(block)

    def _wrap_block_with_checkpoint(self, block):
        """Wrap a block to use gradient checkpointing"""
        original_forward = block.forward

        def checkpointed_forward(x):
            if self.training:
                return checkpoint(original_forward, x, use_reentrant=False)
            else:
                return original_forward(x)

        block.forward = checkpointed_forward
        return block

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def estimate_loss(self, data_loader, eval_iters=20):
        """Estimate loss on evaluation data"""
        self.eval()
        losses = []

        for i, (x, y) in enumerate(data_loader):
            if i >= eval_iters:
                break

            x, y = x.to(self.config.device), y.to(self.config.device)

            if self.config.use_amp:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    _, loss = self(x, y)
            else:
                _, loss = self(x, y)

            losses.append(loss.item())

        self.train()
        return sum(losses) / len(losses) if losses else float("inf")
