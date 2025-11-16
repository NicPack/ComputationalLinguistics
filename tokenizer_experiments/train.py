import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from data_utils import load_datasets
from tqdm import tqdm

import bdh
from config import CONFIG


def train(tokenizer_type: str):
    """
    Train BDH model with specified tokenizer.

    This is a modified version of your original training script that:
    - Uses pre-tokenized data
    - Supports variable vocabulary sizes
    - Tracks training metrics
    - Saves comprehensive checkpoints
    """

    print(f"\n{'=' * 80}")
    print(f"Training BDH model with {tokenizer_type} tokenizer")
    print(f"{'=' * 80}\n")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]

    ctx = (
        torch.amp.autocast(device_type=device.type, dtype=ptdtype)
        if "cuda" in device.type
        else nullcontext()
    )
    scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == "float16"))

    # Set seeds for reproducibility
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Mixed precision: {'enabled' if 'cuda' in device.type else 'disabled'}")

    # Load datasets
    print("\nLoading datasets...")
    train_dataset, test_dataset, vocab_size = load_datasets(tokenizer_type, CONFIG)

    # Initialize model
    print("\nInitializing model...")
    model_config = bdh.BDHConfig(
        n_layer=CONFIG.n_layer,
        n_embd=CONFIG.n_embd,
        dropout=CONFIG.dropout,
        n_head=CONFIG.n_head,
        mlp_internal_dim_multiplier=CONFIG.mlp_internal_dim_multiplier,
        vocab_size=vocab_size,
    )

    model = bdh.BDH(model_config).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Compile model for faster training (PyTorch 2.0+)
    print("Compiling model...")
    model = torch.compile(model)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG.learning_rate,
        weight_decay=CONFIG.weight_decay,
    )

    # Training metrics
    training_metrics = {
        "tokenizer_type": tokenizer_type,
        "vocab_size": vocab_size,
        "n_params": n_params,
        "config": {
            "block_size": CONFIG.block_size,
            "batch_size": CONFIG.batch_size,
            "max_iters": CONFIG.max_iters,
            "learning_rate": CONFIG.learning_rate,
            "weight_decay": CONFIG.weight_decay,
        },
        "losses": [],
        "steps": [],
    }

    # Checkpoint path
    checkpoint_path = Path(CONFIG.checkpoint_dir) / f"bdh_{tokenizer_type}.pt"

    print(f"\n{'=' * 80}")
    print("Starting training")
    print(f"{'=' * 80}\n")

    # Training loop
    model.train()
    loss_acc = 0.0
    loss_steps = 0

    start_time = time.time()

    for step in tqdm(range(CONFIG.max_iters)):
        # Get batch
        x, y = train_dataset.get_batch(CONFIG.batch_size)

        # Forward pass
        with ctx:
            logits, loss = model(x, y)

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Accumulate loss
        loss_acc += loss.item()
        loss_steps += 1

        # Logging
        if (step + 1) % CONFIG.log_freq == 0 or step == 0:
            avg_loss = loss_acc / loss_steps
            elapsed = time.time() - start_time
            tokens_per_sec = (
                (step + 1) * CONFIG.batch_size * CONFIG.block_size / elapsed
            )

            print(
                f"Step {step + 1:4d}/{CONFIG.max_iters} | "
                f"Loss: {avg_loss:.4f} | "
                f"Tokens/sec: {tokens_per_sec:,.0f} | "
                f"Time: {elapsed:.1f}s"
            )

            training_metrics["losses"].append(avg_loss)
            training_metrics["steps"].append(step + 1)

            loss_acc = 0.0
            loss_steps = 0

            # Save checkpoint
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step + 1,
                    "config": model_config,
                },
                checkpoint_path,
            )

    # Final timing
    total_time = time.time() - start_time
    training_metrics["total_training_time_seconds"] = total_time
    training_metrics["tokens_per_second"] = (
        CONFIG.max_iters * CONFIG.batch_size * CONFIG.block_size / total_time
    )

    print(f"\n{'=' * 80}")
    print("Training completed!")
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} minutes)")
    print(f"Throughput: {training_metrics['tokens_per_second']:,.0f} tokens/sec")
    print(f"Final checkpoint: {checkpoint_path}")
    print(f"{'=' * 80}\n")

    # Save training metrics
    metrics_path = Path(CONFIG.results_dir) / f"training_metrics_{tokenizer_type}.json"
    with open(metrics_path, "w") as f:
        json.dump(training_metrics, f, indent=2)

    print(f"Training metrics saved to: {metrics_path}\n")

    return checkpoint_path, training_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train BDH model with specified tokenizer"
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        required=True,
        choices=["bielik", "whitespace", "sentencepiece"],
        help="Type of tokenizer to use",
    )

    args = parser.parse_args()
    train(args.tokenizer_type)


if __name__ == "__main__":
    main()
