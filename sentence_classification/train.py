import json
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from model import GPTForSequenceClassification
from torch.utils.data import DataLoader
from training_helpers import (
    TrainingMetrics,
    evaluate_model,
    train_epoch,
)
from transformers import get_linear_schedule_with_warmup


def train_from_scratch(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    model_config: Dict,
    training_config: Dict,
    save_dir: Path,
    device: str = "cuda",
) -> Tuple[nn.Module, TrainingMetrics, Dict]:
    """
    Train model from scratch

    Args:
        train_dataloader: Training data
        val_dataloader: Validation data
        model_config: Model architecture config
        training_config: Training hyperparameters
        save_dir: Directory to save checkpoints and metrics
        device: Device for training

    Returns:
        trained_model, metrics, timing_info
    """
    print("\n" + "=" * 80)
    print("TRAINING FROM SCRATCH")
    print("=" * 80)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    print("\nInitializing model...")
    model = GPTForSequenceClassification(**model_config).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        betas=(0.9, 0.999),
    )

    # Learning rate scheduler with warmup
    num_training_steps = len(train_dataloader) * training_config["num_epochs"]
    num_warmup_steps = int(num_training_steps * training_config["warmup_ratio"])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    print(f"\nTraining for {training_config['num_epochs']} epochs")
    print(f"Total steps: {num_training_steps}")
    print(f"Warmup steps: {num_warmup_steps}")

    # Metrics tracking
    metrics = TrainingMetrics()
    best_val_f1 = 0
    total_training_time = 0

    # Training loop
    for epoch in range(training_config["num_epochs"]):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch + 1}/{training_config['num_epochs']}")
        print(f"{'=' * 80}")

        epoch_start = time.time()

        # Train
        train_loss, train_acc, train_f1, lrs = train_epoch(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            device,
            max_grad_norm=training_config["max_grad_norm"],
        )

        # Validate
        print("\nValidating...")
        val_loss, val_acc, val_f1, val_report = evaluate_model(
            model, val_dataloader, device
        )

        epoch_time = time.time() - epoch_start
        total_training_time += epoch_time

        # Update metrics
        metrics.update(
            {
                "train_losses": train_loss,
                "val_losses": val_loss,
                "train_accuracies": train_acc,
                "val_accuracies": val_acc,
                "train_f1s": train_f1,
                "val_f1s": val_f1,
                "epoch_times": epoch_time,
            }
        )
        metrics.learning_rates.extend(lrs)

        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(
            f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}"
        )
        print(
            f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
        )
        print(f"  Epoch Time: {epoch_time:.2f}s")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            checkpoint_path = save_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": val_f1,
                    "val_accuracy": val_acc,
                },
                checkpoint_path,
            )
            print(f"  ✓ Saved best model (F1: {val_f1:.4f})")

        # Save checkpoint every N epochs
        if (epoch + 1) % training_config.get("save_every", 5) == 0:
            checkpoint_path = save_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )

    # Save final model
    final_path = save_dir / "final_model.pt"
    torch.save(model.state_dict(), final_path)

    # Save metrics
    metrics.save(save_dir / "training_metrics.json")
    metrics.plot_curves(save_dir / "training_curves.png")

    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    _, final_acc, final_f1, final_report = evaluate_model(model, val_dataloader, device)

    print("\nFinal Validation Metrics:")
    print(f"  Accuracy: {final_acc:.4f}")
    print(f"  F1 Score: {final_f1:.4f}")
    print("\nClassification Report:")
    print(json.dumps(final_report, indent=2))

    # Save classification report
    with open(save_dir / "classification_report.json", "w") as f:
        json.dump(final_report, f, indent=2)

    # Timing info
    timing_info = {
        "total_training_time_seconds": total_training_time,
        "avg_epoch_time_seconds": total_training_time / training_config["num_epochs"],
        "total_training_time_hours": total_training_time / 3600,
    }

    with open(save_dir / "timing_info.json", "w") as f:
        json.dump(timing_info, f, indent=2)

    print(
        f"\nTotal Training Time: {timing_info['total_training_time_hours']:.2f} hours"
    )

    return model, metrics, timing_info


def fine_tune_pretrained(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    pretrained_model_path: Path,
    model_config: Dict,
    training_config: Dict,
    save_dir: Path,
    device: str = "cuda",
    freeze_backbone: bool = False,
) -> Tuple[nn.Module, TrainingMetrics, Dict]:
    """
    Fine-tune pre-trained model

    Args:
        train_dataloader: Training data
        val_dataloader: Validation data
        pretrained_model_path: Path to pre-trained GPT checkpoint
        model_config: Model architecture config
        training_config: Training hyperparameters (should have lower LR)
        save_dir: Directory to save checkpoints and metrics
        device: Device for training
        freeze_backbone: If True, only train classification head

    Returns:
        trained_model, metrics, timing_info
    """
    print("\n" + "=" * 80)
    print("FINE-TUNING PRE-TRAINED MODEL")
    print("=" * 80)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    print("\nInitializing model with pre-trained weights...")
    model = GPTForSequenceClassification(**model_config).to(device)

    # Load pre-trained weights
    print(f"Loading pre-trained weights from: {pretrained_model_path}")
    pretrained_state = torch.load(
        pretrained_model_path, map_location=device, weights_only=False
    )

    # Handle different checkpoint formats
    if "model_state_dict" in pretrained_state:
        pretrained_state = pretrained_state["model_state_dict"]

    # Load only the GPT backbone weights
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v
        for k, v in pretrained_state.items()
        if k in model_dict and "classifier" not in k
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)

    print(f"Loaded {len(pretrained_dict)} pre-trained parameters")

    # Optionally freeze backbone
    if freeze_backbone:
        print("\nFreezing backbone parameters...")
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,}")
    else:
        print(
            f"\nFine-tuning all parameters: {sum(p.numel() for p in model.parameters()):,}"
        )

    # Optimizer (only trainable parameters)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        betas=(0.9, 0.999),
    )

    # Learning rate scheduler with warmup
    num_training_steps = len(train_dataloader) * training_config["num_epochs"]
    num_warmup_steps = int(num_training_steps * training_config["warmup_ratio"])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    print(f"\nFine-tuning for {training_config['num_epochs']} epochs")
    print(f"Total steps: {num_training_steps}")
    print(f"Warmup steps: {num_warmup_steps}")
    print(f"Learning rate: {training_config['learning_rate']}")

    # Metrics tracking
    metrics = TrainingMetrics()
    best_val_f1 = 0
    total_training_time = 0

    # Training loop (same as from-scratch but typically fewer epochs)
    for epoch in range(training_config["num_epochs"]):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch + 1}/{training_config['num_epochs']}")
        print(f"{'=' * 80}")

        epoch_start = time.time()

        # Train
        train_loss, train_acc, train_f1, lrs = train_epoch(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            device,
            max_grad_norm=training_config["max_grad_norm"],
        )

        # Validate
        print("\nValidating...")
        val_loss, val_acc, val_f1, val_report = evaluate_model(
            model, val_dataloader, device
        )

        epoch_time = time.time() - epoch_start
        total_training_time += epoch_time

        # Update metrics
        metrics.update(
            {
                "train_losses": train_loss,
                "val_losses": val_loss,
                "train_accuracies": train_acc,
                "val_accuracies": val_acc,
                "train_f1s": train_f1,
                "val_f1s": val_f1,
                "epoch_times": epoch_time,
            }
        )
        metrics.learning_rates.extend(lrs)

        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(
            f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}"
        )
        print(
            f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
        )
        print(f"  Epoch Time: {epoch_time:.2f}s")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            checkpoint_path = save_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": val_f1,
                    "val_accuracy": val_acc,
                },
                checkpoint_path,
            )
            print(f"  ✓ Saved best model (F1: {val_f1:.4f})")

    # Save final model
    final_path = save_dir / "final_model.pt"
    torch.save(model.state_dict(), final_path)

    # Save metrics
    metrics.save(save_dir / "training_metrics.json")
    metrics.plot_curves(save_dir / "training_curves.png")

    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    _, final_acc, final_f1, final_report = evaluate_model(model, val_dataloader, device)

    print("\nFinal Validation Metrics:")
    print(f"  Accuracy: {final_acc:.4f}")
    print(f"  F1 Score: {final_f1:.4f}")
    print("\nClassification Report:")
    print(json.dumps(final_report, indent=2))

    # Save classification report
    with open(save_dir / "classification_report.json", "w") as f:
        json.dump(final_report, f, indent=2)

    # Timing info
    timing_info = {
        "total_training_time_seconds": total_training_time,
        "avg_epoch_time_seconds": total_training_time / training_config["num_epochs"],
        "total_training_time_hours": total_training_time / 3600,
    }

    with open(save_dir / "timing_info.json", "w") as f:
        json.dump(timing_info, f, indent=2)

    print(
        f"\nTotal Fine-tuning Time: {timing_info['total_training_time_hours']:.2f} hours"
    )

    return model, metrics, timing_info
