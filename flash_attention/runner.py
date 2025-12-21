import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from utils import MemoryTracker

from config import ExperimentConfig
from models import GPTLanguageModel


class ExperimentRunner:
    """Run and track experiments"""

    def __init__(self, results_dir="./experiment_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.memory_tracker = MemoryTracker()

    def run_experiment(self, config: ExperimentConfig, train_dataset, val_dataset):
        """Run a single experiment with given configuration"""
        print(f"\n{'=' * 60}")
        print(f"Running Experiment: {config}")
        print(f"{'=' * 60}")

        # Reset memory tracking
        self.memory_tracker.reset()

        # Initialize model
        model = GPTLanguageModel(config).to(config.device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

        # Data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False
        )

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

        # AMP scaler
        scaler = torch.GradScaler(enabled=config.use_amp)

        # Training metrics
        results = {
            "config": str(config),
            "n_params": n_params,
            "train_losses": [],
            "val_losses": [],
            "timestamps": [],
            "memory_stats": [],
        }

        start_time = time.time()
        best_val_loss = float("inf")

        print("\nTraining started...\n")

        # Training loop
        iter_count = 0
        for epoch in range(100):
            for x, y in train_loader:
                if iter_count >= config.max_iters:
                    break

                x, y = x.to(config.device), y.to(config.device)

                # Forward pass with optional AMP
                optimizer.zero_grad()

                if config.use_amp:
                    with torch.autocast(
                        device_type=config.device, dtype=torch.bfloat16
                    ):
                        logits, loss = model(x, y)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits, loss = model(x, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                # Logging
                if iter_count % config.eval_interval == 0:
                    val_loss = model.estimate_loss(val_loader)
                    elapsed = time.time() - start_time

                    results["train_losses"].append(loss.item())
                    results["val_losses"].append(val_loss)
                    results["timestamps"].append(elapsed)
                    results["memory_stats"].append(
                        self.memory_tracker.get_memory_stats()
                    )

                    print(
                        f"Step {iter_count:4d} | "
                        f"Train Loss: {loss.item():.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Time: {elapsed:.1f}s"
                    )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss

                iter_count += 1

            if iter_count >= config.max_iters:
                break

        # Final evaluation
        final_val_loss = model.estimate_loss(val_loader, eval_iters=50)
        final_train_loss = model.estimate_loss(train_loader, eval_iters=50)
        total_time = time.time() - start_time

        # Final memory stats
        final_memory = self.memory_tracker.get_memory_stats()

        results.update(
            {
                "final_train_loss": final_train_loss,
                "final_val_loss": final_val_loss,
                "final_perplexity": math.exp(final_val_loss),
                "best_val_loss": best_val_loss,
                "total_time_seconds": total_time,
                "time_per_iter": total_time / config.max_iters,
                "final_memory_stats": final_memory,
            }
        )

        print(f"\n{'=' * 60}")
        print("Experiment Complete")
        print(f"{'=' * 60}")
        print(f"Final Training Loss: {final_train_loss:.4f}")
        print(f"Final Validation Loss: {final_val_loss:.4f}")
        print(f"Final Perplexity: {math.exp(final_val_loss):.2f}")
        print(f"Total Time: {total_time:.1f}s")
        print(f"Time per Iteration: {total_time / config.max_iters * 1000:.1f}ms")
        self.memory_tracker.print_memory_summary()

        # Save results
        self.save_results(config, results)

        # Cleanup
        del model, optimizer, scaler
        torch.cuda.empty_cache()

        return results

    def save_results(self, config: ExperimentConfig, results: dict):
        """Save experiment results to file"""
        filename = self.results_dir / f"{config}.json"
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {filename}")
