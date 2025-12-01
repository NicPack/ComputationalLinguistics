import json
import time
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def inference_benchmark(
    model: nn.Module, dataloader: DataLoader, device: str, num_runs: int = 3
) -> Dict:
    """
    Benchmark inference time

    Args:
        model: Trained model
        dataloader: Data to run inference on
        device: Device for inference
        num_runs: Number of runs to average

    Returns:
        Timing statistics
    """
    model.eval()
    inference_times = []

    print(f"\nRunning inference benchmark ({num_runs} runs)...")

    with torch.no_grad():
        for run in range(num_runs):
            start_time = time.time()

            for batch in tqdm(dataloader, desc=f"Run {run + 1}/{num_runs}"):
                input_ids = batch["input_ids"].to(device)
                _ = model(input_ids)

            run_time = time.time() - start_time
            inference_times.append(run_time)

    num_samples = len(dataloader.dataset)

    timing_stats = {
        "total_samples": num_samples,
        "avg_total_time_seconds": np.mean(inference_times),
        "std_total_time_seconds": np.std(inference_times),
        "avg_time_per_sample_ms": (np.mean(inference_times) / num_samples) * 1000,
        "throughput_samples_per_second": num_samples / np.mean(inference_times),
    }

    print("\nInference Timing:")
    print(
        f"  Avg time: {timing_stats['avg_total_time_seconds']:.2f}s ± {timing_stats['std_total_time_seconds']:.2f}s"
    )
    print(f"  Per sample: {timing_stats['avg_time_per_sample_ms']:.2f}ms")
    print(
        f"  Throughput: {timing_stats['throughput_samples_per_second']:.1f} samples/sec"
    )

    return timing_stats


def compare_experiments(from_scratch_dir: Path, fine_tuned_dir: Path, save_dir: Path):
    """
    Compare from-scratch vs fine-tuned results

    Args:
        from_scratch_dir: Directory with from-scratch results
        fine_tuned_dir: Directory with fine-tuned results
        save_dir: Directory to save comparison plots
    """
    print("\n" + "=" * 80)
    print("COMPARING EXPERIMENTS")
    print("=" * 80)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load metrics
    with open(from_scratch_dir / "training_metrics.json") as f:
        scratch_metrics = json.load(f)

    with open(fine_tuned_dir / "training_metrics.json") as f:
        finetuned_metrics = json.load(f)

    # Load timing info
    with open(from_scratch_dir / "timing_info.json") as f:
        scratch_timing = json.load(f)

    with open(fine_tuned_dir / "timing_info.json") as f:
        finetuned_timing = json.load(f)

    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Validation Loss
    axes[0, 0].plot(
        scratch_metrics["val_losses"], label="From Scratch", linewidth=2, marker="o"
    )
    axes[0, 0].plot(
        finetuned_metrics["val_losses"], label="Fine-tuned", linewidth=2, marker="s"
    )
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Validation Loss")
    axes[0, 0].set_title("Validation Loss Comparison")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Validation Accuracy
    axes[0, 1].plot(
        scratch_metrics["val_accuracies"], label="From Scratch", linewidth=2, marker="o"
    )
    axes[0, 1].plot(
        finetuned_metrics["val_accuracies"], label="Fine-tuned", linewidth=2, marker="s"
    )
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Validation Accuracy")
    axes[0, 1].set_title("Validation Accuracy Comparison")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Validation F1
    axes[0, 2].plot(
        scratch_metrics["val_f1s"], label="From Scratch", linewidth=2, marker="o"
    )
    axes[0, 2].plot(
        finetuned_metrics["val_f1s"], label="Fine-tuned", linewidth=2, marker="s"
    )
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Validation F1 Score")
    axes[0, 2].set_title("Validation F1 Score Comparison")
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # Training Loss
    axes[1, 0].plot(
        scratch_metrics["train_losses"], label="From Scratch", linewidth=2, marker="o"
    )
    axes[1, 0].plot(
        finetuned_metrics["train_losses"], label="Fine-tuned", linewidth=2, marker="s"
    )
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Training Loss")
    axes[1, 0].set_title("Training Loss Comparison")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Epoch Time
    axes[1, 1].plot(
        scratch_metrics["epoch_times"], label="From Scratch", linewidth=2, marker="o"
    )
    axes[1, 1].plot(
        finetuned_metrics["epoch_times"], label="Fine-tuned", linewidth=2, marker="s"
    )
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Time (seconds)")
    axes[1, 1].set_title("Epoch Time Comparison")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Summary statistics
    summary_text = f"""
    From Scratch:
    • Final Val Acc: {scratch_metrics["val_accuracies"][-1]:.4f}
    • Final Val F1: {scratch_metrics["val_f1s"][-1]:.4f}
    • Best Val F1: {max(scratch_metrics["val_f1s"]):.4f}
    • Training Time: {scratch_timing["total_training_time_hours"]:.2f}h
    
    Fine-tuned:
    • Final Val Acc: {finetuned_metrics["val_accuracies"][-1]:.4f}
    • Final Val F1: {finetuned_metrics["val_f1s"][-1]:.4f}
    • Best Val F1: {max(finetuned_metrics["val_f1s"]):.4f}
    • Training Time: {finetuned_timing["total_training_time_hours"]:.2f}h
    
    Improvement:
    • Acc: {(finetuned_metrics["val_accuracies"][-1] - scratch_metrics["val_accuracies"][-1]) * 100:+.2f}%
    • F1: {(finetuned_metrics["val_f1s"][-1] - scratch_metrics["val_f1s"][-1]) * 100:+.2f}%
    • Time: {(scratch_timing["total_training_time_hours"] - finetuned_timing["total_training_time_hours"]):.2f}h saved
    """

    axes[1, 2].text(
        0.1,
        0.5,
        summary_text,
        fontsize=10,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_dir / "comparison_plots.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Save comparison summary
    comparison = {
        "from_scratch": {
            "final_val_accuracy": scratch_metrics["val_accuracies"][-1],
            "final_val_f1": scratch_metrics["val_f1s"][-1],
            "best_val_f1": max(scratch_metrics["val_f1s"]),
            "training_time_hours": scratch_timing["total_training_time_hours"],
        },
        "fine_tuned": {
            "final_val_accuracy": finetuned_metrics["val_accuracies"][-1],
            "final_val_f1": finetuned_metrics["val_f1s"][-1],
            "best_val_f1": max(finetuned_metrics["val_f1s"]),
            "training_time_hours": finetuned_timing["total_training_time_hours"],
        },
        "improvements": {
            "accuracy_gain": finetuned_metrics["val_accuracies"][-1]
            - scratch_metrics["val_accuracies"][-1],
            "f1_gain": finetuned_metrics["val_f1s"][-1]
            - scratch_metrics["val_f1s"][-1],
            "time_saved_hours": scratch_timing["total_training_time_hours"]
            - finetuned_timing["total_training_time_hours"],
        },
    }

    with open(save_dir / "comparison_summary.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + summary_text)
    print(f"\nComparison plots saved to: {save_dir / 'comparison_plots.png'}")
