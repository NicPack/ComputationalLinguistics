import dataclasses
from pathlib import Path


@dataclasses.dataclass
class ExperimentConfig:
    """Shared configuration across all tokenizer experiments."""

    # Data paths
    raw_data_path: str = "../datasets/high_quality_plwiki.txt"
    tokenized_data_dir: str = "datasets/tokenized"
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"

    # Tokenizer paths
    bielik_tokenizer_path: str = "../tokenizers/bielik_tokenizer"
    whitespace_tokenizer_path: str = "../tokenizers/whitespace_tokenizer.json"
    sentencepiece_model_path: str = "../tokenizers/sentencePieceTokenizer.model"

    # Training hyperparameters (match your original setup)
    block_size: int = 512
    batch_size: int = 16
    max_iters: int = 3000
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    log_freq: int = 100

    # Model architecture (BDH)
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128

    # Data splits
    train_ratio: float = 0.9

    # Evaluation
    eval_batch_size: int = 16
    eval_iters: int = 100

    # Generation
    generation_prompts: list = dataclasses.field(
        default_factory=lambda: [
            "Warszawa to stolica",
            "Historia Polski rozpoczęła się",
            "Największą rzeką w Polsce jest",
            "W roku 1945",
            "Polscy naukowcy odkryli",
            "Kraków jest miastem",
            "Literatura polska słynie z",
            "Podczas II wojny światowej",
            "Polska kuchnia jest znana z",
            "Jan Paweł II był",
        ]
    )
    generation_max_tokens: int = 100
    generation_top_k: int = 3

    # Device configuration
    device_type: str = "cuda"  # "cuda", "cpu", or "mps"
    dtype: str = "bfloat16"  # "float32", "bfloat16", or "float16"

    def __post_init__(self):
        """Create necessary directories."""
        Path(self.tokenized_data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)


# Global config instance
CONFIG = ExperimentConfig()
