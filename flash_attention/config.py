from dataclasses import dataclass

import torch

try:
    from flash_attn import flash_attn_func, flash_attn_qkvpacked_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print(
        "flash-attn not available. Install with: pip install flash-attn --no-build-isolation"
    )


@dataclass
class ExperimentConfig:
    """Configuration for experimental setup"""

    # Model architecture
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    block_size: int = 512
    vocab_size: int = 50257
    dropout: float = 0.1

    # Training
    batch_size: int = 8
    learning_rate: float = 3e-4
    max_iters: int = 1000
    eval_interval: int = 100

    # Optimization technique
    use_amp: bool = False
    use_flash_attention: bool = False
    use_local_attention: bool = False
    local_attention_window: int = 128
    use_gradient_checkpointing: bool = False

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.use_flash_attention and not FLASH_ATTN_AVAILABLE:
            raise RuntimeError(
                "FlashAttention requested but flash-attn is not installed. "
            )

        if self.use_local_attention and not FLASH_ATTN_AVAILABLE:
            raise RuntimeError(
                "Local Attention requested but flash-attn is not installed. "
            )

        if (self.use_flash_attention or self.use_local_attention) and not self.use_amp:
            print(
                "FlashAttention/LocalAttention requires BF16. Enabling AMP automatically."
            )
            self.use_amp = True

    def __str__(self):
        opts = []
        if self.use_amp:
            opts.append("BF16-AMP")
        if self.use_flash_attention:
            opts.append("FlashAttn")
        if self.use_local_attention:
            opts.append(f"LocalAttn-W{self.local_attention_window}")
        if self.use_gradient_checkpointing:
            opts.append("GradCP")
        return f"BS{self.batch_size}_" + ("_".join(opts) if opts else "Baseline")
