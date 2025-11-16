from pathlib import Path

import numpy as np
import torch


class TokenizedDataset:
    """Memory-mapped dataset for efficient loading of pre-tokenized data."""
    
    def __init__(self, data_path: str, block_size: int, device: torch.device):
        """
        Args:
            data_path: Path to .npy file containing tokenized data
            block_size: Sequence length for training
            device: torch device for tensors
        """
        if not Path(data_path).exists():
            raise FileNotFoundError(
                f"Tokenized data not found: {data_path}\n"
                f"Did you run prepare_data.py first?"
            )
        
        # Load as memory-mapped array for efficient access
        self.data = np.load(data_path, mmap_mode='r')
        self.block_size = block_size
        self.device = device
        
        if len(self.data) <= block_size:
            raise ValueError(
                f"Dataset too short ({len(self.data)} tokens). "
                f"Need at least {block_size + 1} tokens."
            )
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def get_batch(self, batch_size: int):
        """
        Sample a random batch of sequences.
        
        Returns:
            x: Input sequences [batch_size, block_size]
            y: Target sequences [batch_size, block_size] (shifted by 1)
        """
        # Sample random starting indices
        ix = torch.randint(len(self) - 1, (batch_size,))
        
        # Extract sequences
        x = torch.stack([
            torch.from_numpy(
                self.data[i:i + self.block_size].astype(np.int64)
            )
            for i in ix
        ])
        
        y = torch.stack([
            torch.from_numpy(
                self.data[i + 1:i + 1 + self.block_size].astype(np.int64)
            )
            for i in ix
        ])
        
        # Move to device
        if self.device.type == 'cuda':
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        
        return x, y


def load_datasets(tokenizer_type: str, config):
    """
    Load train and test datasets for a given tokenizer.
    
    Returns:
        train_dataset: TokenizedDataset for training
        test_dataset: TokenizedDataset for evaluation
        vocab_size: Size of vocabulary
    """
    from config import CONFIG
    import json
    
    tokenized_dir = Path(CONFIG.tokenized_data_dir) / tokenizer_type
    
    # Load metadata to get vocab size
    meta_path = tokenized_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {meta_path}\n"
            f"Did you run prepare_data.py for {tokenizer_type}?"
        )
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    vocab_size = meta["vocab_size"]
    
    # Setup device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Load datasets
    train_dataset = TokenizedDataset(
        str(tokenized_dir / "train.npy"),
        config.block_size,
        device
    )
    
    test_dataset = TokenizedDataset(
        str(tokenized_dir / "test.npy"),
        config.block_size,
        device
    )
    
    print(f"Loaded {tokenizer_type} datasets:")
    print(f"  Vocabulary size: {vocab_size:,}")
    print(f"  Train tokens: {len(train_dataset.data):,}")
    print(f"  Test tokens: {len(test_dataset.data):,}")
    
    return train_dataset, test_dataset, vocab_size
