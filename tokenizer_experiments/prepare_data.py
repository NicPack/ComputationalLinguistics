import argparse
import json
import time
from pathlib import Path

import numpy as np
from tokenizer_utils import load_tokenizer
from tqdm import tqdm

from config import CONFIG


def prepare_tokenized_data(tokenizer_type: str):
    """
    Pre-tokenize the entire dataset and save to disk.

    This function:
    1. Loads the raw text file
    2. Tokenizes it using the specified tokenizer
    3. Splits into train/test
    4. Saves as memory-mapped numpy arrays for fast loading
    """

    print(f"\n{'=' * 80}")
    print(f"Preparing data for tokenizer: {tokenizer_type}")
    print(f"{'=' * 80}\n")

    # Load tokenizer
    print(f"Loading {tokenizer_type} tokenizer...")
    tokenizer = load_tokenizer(tokenizer_type, CONFIG)
    vocab_size = tokenizer.vocab_size()
    print(f"Vocabulary size: {vocab_size:,}")

    # Read raw text
    print(f"\nReading raw text from {CONFIG.raw_data_path}...")
    if not Path(CONFIG.raw_data_path).exists():
        raise FileNotFoundError(f"Raw data file not found: {CONFIG.raw_data_path}")

    with open(CONFIG.raw_data_path, "r", encoding="utf-8") as f:
        text = f.read()

    text_size_mb = len(text.encode("utf-8")) / (1024 * 1024)
    print(f"Text size: {text_size_mb:.2f} MB")
    print(f"Characters: {len(text):,}")

    # Tokenize in chunks for efficiency (critical for large files)
    print("\nTokenizing text in chunks...")
    start_time = time.time()

    chunk_size = 1_000_000  # 1MB chunks
    token_ids = []

    num_chunks = (len(text) + chunk_size - 1) // chunk_size

    for i in tqdm(range(0, len(text), chunk_size), total=num_chunks, desc="Tokenizing"):
        chunk = text[i : i + chunk_size]
        chunk_tokens = tokenizer.encode(chunk)
        token_ids.extend(chunk_tokens)

    tokenization_time = time.time() - start_time

    print(f"Tokenization completed in {tokenization_time:.2f} seconds")
    print(f"Total tokens: {len(token_ids):,}")
    print(f"Tokens per character: {len(token_ids) / len(text):.4f}")
    print(f"Throughput: {len(token_ids) / tokenization_time:.0f} tokens/sec")

    # Convert to numpy array
    token_ids_array = np.array(token_ids, dtype=np.int32)

    # Split into train/test
    split_idx = int(CONFIG.train_ratio * len(token_ids_array))
    train_data = token_ids_array[:split_idx]
    test_data = token_ids_array[split_idx:]

    print("\nData split:")
    print(
        f"  Train tokens: {len(train_data):,} ({len(train_data) / len(token_ids_array) * 100:.1f}%)"
    )
    print(
        f"  Test tokens:  {len(test_data):,} ({len(test_data) / len(token_ids_array) * 100:.1f}%)"
    )

    # Save to disk
    output_dir = Path(CONFIG.tokenized_data_dir) / tokenizer_type
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.npy"
    test_path = output_dir / "test.npy"
    meta_path = output_dir / "meta.json"

    print("\nSaving tokenized data...")
    np.save(train_path, train_data)
    np.save(test_path, test_data)

    # Save metadata
    meta = {
        "tokenizer_type": tokenizer_type,
        "vocab_size": vocab_size,
        "total_tokens": len(token_ids),
        "train_tokens": len(train_data),
        "test_tokens": len(test_data),
        "text_size_bytes": len(text.encode("utf-8")),
        "text_characters": len(text),
        "tokenization_time_seconds": tokenization_time,
        "tokens_per_character": len(token_ids) / len(text),
        "train_ratio": CONFIG.train_ratio,
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Train data: {train_path}")
    print(f"  Test data:  {test_path}")
    print(f"  Metadata:   {meta_path}")

    print(f"\n{'=' * 80}")
    print(f"Data preparation complete for {tokenizer_type}!")
    print(f"{'=' * 80}\n")

    return meta


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize dataset for training")
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        required=True,
        choices=["bielik", "whitespace", "sentencepiece"],
        help="Type of tokenizer to use",
    )

    args = parser.parse_args()
    prepare_tokenized_data(args.tokenizer_type)


if __name__ == "__main__":
    main()
