import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Dataset for character-level or BPE tokenized text"""

    def __init__(self, data, block_size):
        """
        Args:
            data: torch.tensor of token indices
            block_size: context length
        """
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Get a chunk of block_size tokens
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def load_data(data_path, vocab_size=None, encoding=None):
    """
    Load and prepare text data for training

    Args:
        data_path: path to text file
        vocab_size: if provided, will use simple character encoding
        encoding: if provided, will use this encoding (e.g., tiktoken)

    Returns:
        train_data, val_data, vocab_size, encode, decode functions
    """
    print(f"\nLoading data from {data_path}...")

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"   Text length: {len(text):,} characters")

    # If no encoding provided, use character-level encoding
    if encoding is None:
        print("   Using character-level encoding...")
        chars = sorted(list(set(text)))
        vocab_size = len(chars)

        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}

        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: "".join([itos[i] for i in l])

        print(f"Vocabulary size: {vocab_size}")
    else:
        print("Using provided encoding...")
        encode = encoding.encode
        decode = encoding.decode
        if vocab_size is None:
            # Try to get vocab size from encoding
            vocab_size = getattr(
                encoding, "n_vocab", 50257
            )  # default to GPT-2 vocab size

    # Encode entire text
    data = torch.tensor(encode(text), dtype=torch.long)
    print(f"   Encoded data length: {len(data):,} tokens")

    # Train/val split
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens: {len(val_data):,}")

    return train_data, val_data, vocab_size, encode, decode


class SimpleTokenizer:
    """Simple character-level tokenizer"""

    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return "".join([self.itos[i] for i in l])
