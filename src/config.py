import tiktoken
import torch
from pydantic_settings import BaseSettings
from typing import Any


class Settings(BaseSettings):
    model_name: str = "LSTM"
    dataset_name: str = "plwiki"
    train_split: float = 0.95
    val_split: float = 0.025
    test_split: float = 0.025

    # Tokenization
    @property
    def vocab_size(self):
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        return encoding.n_vocab

    max_seq_length: int = 512

    # Model Architecture
    batch_size: int = 64  # how many independent sequences will we process in parallel?
    block_size: int = 256  # what is the maximum context length for predictions?
    max_iters: int = 5000
    eval_interval: int = 500
    learning_rate: float = 3e-4
    eval_iters: int = 200
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2

    # Training
    batch_size: int = 24
    learning_rate: float = 3e-4
    num_epochs: int = 100
    gradient_clip: float = 1.0
    optimizer: Any = torch.optim.AdamW

    # Backend
    if torch.backends.mps.is_available():
        device: str = "mps"
    elif torch.cuda.is_available():
        device: str = "cuda"
    else:
        device: str = "cpu"

    # Evaluation
    eval_batch_size: int = 64
    eval_interval: int = 10

    # Checkpoints
    checkpoint_dir: str = "checkpoints"

    # Weights & Biases
    use_wandb: bool = True
    wandb_project: str = "lingwistyka-lab1"
    wandb_entity: str = "stunick-agh"
