import argparse
import json
import math
import time
from pathlib import Path

import torch
from data_utils import load_datasets
from tokenizer_utils import load_tokenizer

import bdh
from config import CONFIG


@torch.no_grad()
def compute_perplexity(model, dataset, config, num_batches=None):
    """
    Compute perplexity on a dataset.

    Returns:
        perplexity: exp(average loss)
        avg_loss: average cross-entropy loss
    """
    model.eval()

    if num_batches is None:
        num_batches = min(config.eval_iters, len(dataset) // config.eval_batch_size)

    total_loss = 0.0
    total_tokens = 0

    for _ in range(num_batches):
        x, y = dataset.get_batch(config.eval_batch_size)
        logits, loss = model(x, y)

        batch_tokens = x.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity, avg_loss


def compute_character_perplexity(token_perplexity, avg_tokens_per_char):
    """
    Convert token-level perplexity to character-level perplexity.

    Formula: PPL_char = PPL_token ^ (1 / avg_tokens_per_char)

    Reasoning: If on average we need N tokens to represent one character,
    then the per-character uncertainty is the N-th root of per-token uncertainty.
    """
    return token_perplexity ** (1.0 / avg_tokens_per_char)


def compute_word_perplexity(model, tokenizer, test_text, config, device):
    """
    Compute word-level perplexity by evaluating on word boundaries.

    This splits text into words, computes loss per word, and averages.
    """
    model.eval()

    # Split into words (simple whitespace splitting)
    words = test_text.split()

    # We'll compute loss for each word independently
    # This is computationally expensive but gives true word-level perplexity
    word_losses = []

    # Process in chunks to avoid OOM
    chunk_size = 100

    for i in range(0, len(words), chunk_size):
        chunk_words = words[i : i + chunk_size]

        for word in chunk_words:
            # Add space before word (except first word)
            word_text = " " + word if word != words[0] else word

            try:
                # Encode word
                token_ids = tokenizer.encode(word_text)

                if len(token_ids) == 0:
                    continue

                # Create input/target tensors
                if len(token_ids) == 1:
                    # Single token word - use a dummy context
                    x = torch.tensor([[0]], dtype=torch.long, device=device)
                    y = torch.tensor([[token_ids[0]]], dtype=torch.long, device=device)
                else:
                    x = torch.tensor([token_ids[:-1]], dtype=torch.long, device=device)
                    y = torch.tensor([token_ids[1:]], dtype=torch.long, device=device)

                # Compute loss
                logits, loss = model(x, y)
                word_losses.append(loss.item())

            except Exception:
                # Skip problematic words
                continue

    if len(word_losses) == 0:
        return float("inf"), float("inf")

    avg_loss = sum(word_losses) / len(word_losses)
    perplexity = math.exp(avg_loss)

    return perplexity, avg_loss


def compute_oov_stats(tokenizer, test_text, tokenizer_type):
    """
    Compute OOV statistics for whitespace tokenizer.

    For other tokenizers, this metric is less meaningful since they use
    subword tokenization.
    """
    if tokenizer_type != "whitespace":
        return {
            "note": f"OOV not computed for {tokenizer_type} (uses subword tokenization)"
        }

    words = test_text.split()
    unique_words = set(words)

    # Check which words are OOV
    oov_words = set()
    for word in unique_words:
        tokens = tokenizer.tokenize(word)
        # If word is split into multiple tokens or is UNK, it's OOV
        if len(tokens) > 1 or any(t in ["<unk>", "[UNK]", "<UNK>"] for t in tokens):
            oov_words.add(word)

    # Count occurrences
    total_word_occurrences = len(words)
    oov_occurrences = sum(1 for w in words if w in oov_words)

    return {
        "total_words": total_word_occurrences,
        "unique_words": len(unique_words),
        "oov_unique_words": len(oov_words),
        "oov_word_occurrences": oov_occurrences,
        "oov_rate_unique": len(oov_words) / len(unique_words) if unique_words else 0,
        "oov_rate_total": oov_occurrences / total_word_occurrences
        if total_word_occurrences
        else 0,
    }


def compute_tokenizer_efficiency(tokenizer, test_text):
    """
    Compute tokenizer efficiency metrics:
    - Average tokens per word
    - Compression ratio (chars per token)
    - Words directly in vocabulary
    """
    words = test_text.split()

    total_tokens = 0
    total_chars = len(test_text)
    words_as_single_token = 0

    for word in words:
        tokens = tokenizer.tokenize(word)
        total_tokens += len(tokens)

        if len(tokens) == 1 and word.strip() == tokens[0].strip():
            words_as_single_token += 1

    return {
        "total_characters": total_chars,
        "total_tokens": total_tokens,
        "total_words": len(words),
        "avg_tokens_per_word": total_tokens / len(words) if words else 0,
        "avg_chars_per_token": total_chars / total_tokens if total_tokens else 0,
        "words_as_single_token": words_as_single_token,
        "pct_words_as_single_token": words_as_single_token / len(words) * 100
        if words
        else 0,
    }


def measure_inference_time(model, dataset, config, num_batches=50):
    """Measure inference throughput (tokens per second)."""
    model.eval()

    # Warmup
    for _ in range(5):
        x, y = dataset.get_batch(config.eval_batch_size)
        _ = model(x, y)

    # Measure
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()

    total_tokens = 0
    for _ in range(num_batches):
        x, y = dataset.get_batch(config.eval_batch_size)
        _ = model(x, y)
        total_tokens += x.numel()

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start_time

    tokens_per_sec = total_tokens / elapsed

    return tokens_per_sec, elapsed


def evaluate(tokenizer_type: str):
    """
    Comprehensive evaluation of trained model.
    """
    print(f"\n{'=' * 80}")
    print(f"Evaluating {tokenizer_type} model")
    print(f"{'=' * 80}\n")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer(tokenizer_type, CONFIG)

    # Load datasets
    print("Loading datasets...")
    train_dataset, test_dataset, vocab_size = load_datasets(tokenizer_type, CONFIG)

    # Load model
    print("Loading trained model...")
    checkpoint_path = Path(CONFIG.checkpoint_dir) / f"bdh_{tokenizer_type}.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Did you run train.py for {tokenizer_type}?"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint["config"]

    model = bdh.BDH(model_config).to(device)

    # Handle torch.compile wrapper prefix
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        # Strip _orig_mod. prefix added by torch.compile
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    print(f"Model loaded from step {checkpoint['step']}")

    # Get test text for word-level metrics
    print("\nLoading test text...")
    test_text_tokens = test_dataset.data[:1_000_000]  # Use 1M tokens for efficiency
    test_text = tokenizer.decode(test_text_tokens.tolist())

    print(f"Test text: {len(test_text):,} characters, {len(test_text.split()):,} words")

    # === PERPLEXITY EVALUATION ===
    print(f"\n{'=' * 80}")
    print("Computing Perplexity")
    print(f"{'=' * 80}\n")

    print("Token-level perplexity...")
    token_ppl, token_loss = compute_perplexity(model, test_dataset, CONFIG)
    print(f"  Token PPL: {token_ppl:.2f}")
    print(f"  Token Loss: {token_loss:.4f}")

    print("\nComputing tokenizer efficiency for character-level PPL...")
    efficiency = compute_tokenizer_efficiency(tokenizer, test_text)
    avg_tokens_per_char = 1.0 / efficiency["avg_chars_per_token"]

    print(f"  Avg tokens per character: {avg_tokens_per_char:.4f}")

    char_ppl = compute_character_perplexity(token_ppl, avg_tokens_per_char)
    print(f"\nCharacter-level perplexity: {char_ppl:.2f}")

    print("\nComputing word-level perplexity (this may take a few minutes)...")
    word_ppl, word_loss = compute_word_perplexity(
        model, tokenizer, test_text, CONFIG, device
    )
    print(f"Word-level perplexity: {word_ppl:.2f}")

    # === OOV STATISTICS ===
    print(f"\n{'=' * 80}")
    print("Computing OOV Statistics")
    print(f"{'=' * 80}\n")

    oov_stats = compute_oov_stats(tokenizer, test_text, tokenizer_type)
    if "note" in oov_stats:
        print(oov_stats["note"])
    else:
        print(f"Total words: {oov_stats['total_words']:,}")
        print(f"Unique words: {oov_stats['unique_words']:,}")
        print(f"OOV unique words: {oov_stats['oov_unique_words']:,}")
        print(f"OOV rate (unique): {oov_stats['oov_rate_unique'] * 100:.2f}%")
        print(f"OOV rate (total): {oov_stats['oov_rate_total'] * 100:.2f}%")

    # === EFFICIENCY METRICS ===
    print(f"\n{'=' * 80}")
    print("Computing Efficiency Metrics")
    print(f"{'=' * 80}\n")

    print("Tokenizer efficiency (on 1MB test subset):")
    print(f"  Characters: {efficiency['total_characters']:,}")
    print(f"  Tokens: {efficiency['total_tokens']:,}")
    print(f"  Words: {efficiency['total_words']:,}")
    print(f"  Avg tokens per word: {efficiency['avg_tokens_per_word']:.3f}")
    print(f"  Avg characters per token: {efficiency['avg_chars_per_token']:.3f}")
    print(
        f"  Words as single token: {efficiency['words_as_single_token']:,} ({efficiency['pct_words_as_single_token']:.1f}%)"
    )

    print("\nMeasuring inference speed...")
    tokens_per_sec, elapsed = measure_inference_time(model, test_dataset, CONFIG)
    print(f"  Inference: {tokens_per_sec:,.0f} tokens/sec")
    print(
        f"  Time: {elapsed:.2f}s for {CONFIG.eval_batch_size * 50 * CONFIG.block_size:,} tokens"
    )

    # === COMPILE RESULTS ===
    results = {
        "tokenizer_type": tokenizer_type,
        "vocab_size": vocab_size,
        "perplexity": {
            "token_level": token_ppl,
            "character_level": char_ppl,
            "word_level": word_ppl,
        },
        "losses": {
            "token_level": token_loss,
            "word_level": word_loss,
        },
        "oov_statistics": oov_stats,
        "efficiency": {
            **efficiency,
            "inference_tokens_per_sec": tokens_per_sec,
        },
    }

    # Save results
    results_path = Path(CONFIG.results_dir) / f"evaluation_{tokenizer_type}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print("Evaluation complete!")
    print(f"Results saved to: {results_path}")
    print(f"{'=' * 80}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained BDH model")
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        required=True,
        choices=["bielik", "whitespace", "sentencepiece"],
        help="Type of tokenizer to use",
    )

    args = parser.parse_args()
    evaluate(args.tokenizer_type)


if __name__ == "__main__":
    main()
