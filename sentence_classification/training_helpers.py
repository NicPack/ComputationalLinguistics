import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm


class TrainingMetrics:
    """Track and store training metrics"""

    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_f1s = []
        self.val_f1s = []
        self.learning_rates = []
        self.epoch_times = []

    def update(self, metrics: Dict):
        """Update metrics"""
        for key, value in metrics.items():
            if hasattr(self, key):
                getattr(self, key).append(value)

    def save(self, filepath: Path):
        """Save metrics to JSON"""
        data = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
            "train_f1s": self.train_f1s,
            "val_f1s": self.val_f1s,
            "learning_rates": self.learning_rates,
            "epoch_times": self.epoch_times,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def plot_curves(self, save_path: Path):
        """Generate and save training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curves
        axes[0, 0].plot(self.train_losses, label="Train Loss", linewidth=2)
        axes[0, 0].plot(self.val_losses, label="Val Loss", linewidth=2)
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training and Validation Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy curves
        axes[0, 1].plot(self.train_accuracies, label="Train Accuracy", linewidth=2)
        axes[0, 1].plot(self.val_accuracies, label="Val Accuracy", linewidth=2)
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].set_title("Training and Validation Accuracy")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # F1 Score curves
        axes[1, 0].plot(self.train_f1s, label="Train F1", linewidth=2)
        axes[1, 0].plot(self.val_f1s, label="Val F1", linewidth=2)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("F1 Score")
        axes[1, 0].set_title("Training and Validation F1 Score")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Learning rate
        axes[1, 1].plot(self.learning_rates, linewidth=2)
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_ylabel("Learning Rate")
        axes[1, 1].set_title("Learning Rate Schedule")
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


def evaluate_model(
    model: nn.Module, dataloader: DataLoader, device: str
) -> Tuple[float, float, float, Dict]:
    """
    Evaluate model on a dataset

    Returns:
        loss, accuracy, f1_score, classification_report
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = torch.tensor(batch["input_ids"]).to(device)
            labels = torch.tensor(batch["label"]).to(device)

            logits, loss = model(input_ids, labels)

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average="weighted")
    report = classification_report(all_labels, all_predictions, output_dict=True)

    return avg_loss, accuracy, f1, report


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: str,
    max_grad_norm: float = 1.0,
) -> Tuple[float, float, float, List[float]]:
    """
    Train for one epoch

    Returns:
        avg_loss, accuracy, f1_score, learning_rates
    """
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    learning_rates = []

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        labels = torch.tensor(batch["label"]).to(device)

        # Forward pass
        logits, loss = model(input_ids, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        scheduler.step()

        # Track metrics
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=-1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        learning_rates.append(scheduler.get_last_lr()[0])

        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average="weighted")

    return avg_loss, accuracy, f1, learning_rates
