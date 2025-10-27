import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size: int, block_size: int, n_embd: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, block_size, head_size, n_embd, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                Head(head_size, block_size=block_size, n_embd=n_embd, dropout=dropout)
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

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
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, block_size, n_head, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(
            num_heads=n_head,
            block_size=block_size,
            head_size=head_size,
            n_embd=n_embd,
            dropout=dropout,
        )
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(
        self, n_head, block_size, n_embd, n_layer, vocab_size, dropout, device
    ):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd=n_embd, block_size=block_size, n_head=n_head, dropout=dropout
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(normalized_shape=n_embd)  # final layer norm
        self.lm_head = nn.Linear(in_features=n_embd, out_features=vocab_size)
        self.device = device
        self.block_size = block_size
        self.dropout = dropout

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size :]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


class LSTMLanguageModel(nn.Module):
    def __init__(self, n_layer, block_size, n_embd, vocab_size, dropout, device):
        """
        LSTM-based language model for causal language modeling.

        Args:
            n_layer: Number of LSTM layers
            block_size: Maximum sequence length (context window)
            n_embd: Embedding dimension (hidden size)
            vocab_size: Size of vocabulary
            dropout: Dropout probability
            device: Device to run the model on
        """
        super().__init__()

        # Token embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=n_embd,
            hidden_size=n_embd,
            num_layers=n_layer,
            dropout=dropout if n_layer > 1 else 0.0,  # Dropout between LSTM layers
            batch_first=True,
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Layer normalization (optional, but helps training)
        self.ln_f = nn.LayerNorm(n_embd)

        # Output projection to vocabulary
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Save config
        self.device = device
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.vocab_size = vocab_size

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with Xavier/Glorot initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LSTM):
            # Initialize LSTM weights
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    torch.nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    torch.nn.init.zeros_(param.data)

    def forward(self, idx, targets=None, hidden=None):
        """
        Forward pass through the model.

        Args:
            idx: Input token indices (B, T)
            targets: Target token indices (B, T) for computing loss
            hidden: Hidden state tuple (h, c) for LSTM (optional)

        Returns:
            logits: Output logits (B, T, vocab_size)
            loss: Cross-entropy loss (if targets provided)
        """
        B, T = idx.shape

        # Token embeddings: (B, T) -> (B, T, n_embd)
        tok_emb = self.token_embedding_table(idx)

        # Apply dropout to embeddings
        tok_emb = self.dropout(tok_emb)

        # LSTM forward pass: (B, T, n_embd) -> (B, T, n_embd)
        if hidden is None:
            lstm_out, hidden = self.lstm(tok_emb)
        else:
            lstm_out, hidden = self.lstm(tok_emb, hidden)

        # Apply dropout
        lstm_out = self.dropout(lstm_out)

        # Layer normalization
        lstm_out = self.ln_f(lstm_out)

        # Project to vocabulary: (B, T, n_embd) -> (B, T, vocab_size)
        logits = self.lm_head(lstm_out)

        # Compute loss if targets provided
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively.

        Args:
            idx: Starting token indices (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens

        Returns:
            idx: Extended sequence with generated tokens (B, T + max_new_tokens)
        """
        self.eval()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop context to block_size
                idx_cond = (
                    idx
                    if idx.size(1) <= self.block_size
                    else idx[:, -self.block_size :]
                )

                # Get predictions
                logits, _ = self(idx_cond)

                # Focus on last time step
                logits = logits[:, -1, :] / temperature  # (B, vocab_size)

                # Optional top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                # Apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1)  # (B, vocab_size)

                # Sample from distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

                # Append to sequence
                idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        self.train()
        return idx

    def get_num_params(self):
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
