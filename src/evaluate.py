import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import tiktoken
import torch

from config import Settings
from data import estimate_loss
from models import GPTLanguageModel, LSTMLanguageModel


@dataclass
class GenerationResult:
    """Results from a single generation."""

    prompt: str
    generated_text: str
    tokens_generated: int
    inference_time: float
    tokens_per_second: float


@dataclass
class ModelMetrics:
    """Complete metrics for a model."""

    model_name: str
    train_loss: float
    val_loss: float
    train_perplexity: float
    val_perplexity: float
    total_tokens_generated: int
    total_inference_time: float
    mean_time_per_token: float
    generations: List[GenerationResult]


class ModelComparator:
    """Compare Transformer and LSTM models."""

    def __init__(
        self,
        settings: Settings,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        gpt_checkpoint: str,
        lstm_checkpoint: str,
        output_dir: str = "comparison_results",
    ):
        """
        Initialize comparator.

        Args:
            settings: Configuration settings
            train_data: Training data tensor
            val_data: Validation data tensor
            gpt_checkpoint: Path to GPT model checkpoint
            lstm_checkpoint: Path to LSTM model checkpoint
            output_dir: Directory to save results
        """
        self.settings = settings
        self.train_data = train_data
        self.val_data = val_data
        self.gpt_checkpoint = gpt_checkpoint
        self.lstm_checkpoint = lstm_checkpoint
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize tokenizer
        self.encoding = tiktoken.encoding_for_model("gpt-4o-mini")

        # Load models
        print("Loading models...")
        self.gpt_model = self._load_gpt_model()
        self.lstm_model = self._load_lstm_model()
        print("Models loaded successfully!")

    def _load_gpt_model(self) -> GPTLanguageModel:
        """Load GPT (Transformer) model from checkpoint."""
        model = GPTLanguageModel(
            n_head=self.settings.n_head,
            block_size=self.settings.block_size,
            n_embd=self.settings.n_embd,
            n_layer=self.settings.n_layer,
            vocab_size=self.settings.vocab_size,
            dropout=self.settings.dropout,
            device=self.settings.device,
        )
        model.load_state_dict(
            torch.load(
                self.gpt_checkpoint,
                map_location=self.settings.device,
                weights_only=True,
            )
        )
        model = model.to(self.settings.device)
        model.eval()
        return model

    def _load_lstm_model(self) -> LSTMLanguageModel:
        """Load LSTM model from checkpoint."""
        model = LSTMLanguageModel(
            n_layer=self.settings.n_layer,
            block_size=self.settings.block_size,
            n_embd=self.settings.n_embd,
            vocab_size=self.settings.vocab_size,
            dropout=self.settings.dropout,
            device=self.settings.device,
        )
        model.load_state_dict(
            torch.load(
                self.lstm_checkpoint,
                map_location=self.settings.device,
                weights_only=True,
            )
        )
        model = model.to(self.settings.device)
        model.eval()
        return model

    def evaluate_losses(self, model) -> Tuple[float, float]:
        """
        Evaluate train and validation losses.

        Returns:
            Tuple of (train_loss, val_loss)
        """
        print("Evaluating losses...")
        losses = estimate_loss(
            model=model,
            eval_iters=self.settings.eval_iters,
            block_size=self.settings.block_size,
            batch_size=self.settings.batch_size,
            train_data=self.train_data,
            val_data=self.val_data,
            device=self.settings.device,
        )
        return losses["train"], losses["val"]

    def generate_from_prompt(
        self, model, prompt: str, max_new_tokens: int = 100
    ) -> GenerationResult:
        """
        Generate text from a prompt and measure time.

        Args:
            model: Model to use for generation
            prompt: Input prompt text
            max_new_tokens: Maximum tokens to generate

        Returns:
            GenerationResult with metrics
        """
        # Encode prompt
        prompt_tokens = self.encoding.encode(prompt)
        context = torch.tensor(
            [prompt_tokens], dtype=torch.long, device=self.settings.device
        )

        # Generate with timing
        start_time = time.time()
        with torch.no_grad():
            generated_tokens = model.generate(context, max_new_tokens=max_new_tokens)
        inference_time = time.time() - start_time

        # Decode
        generated_text = self.encoding.decode(generated_tokens[0].tolist())

        # Calculate metrics
        tokens_generated = generated_tokens.shape[1] - context.shape[1]
        tokens_per_second = (
            tokens_generated / inference_time if inference_time > 0 else 0
        )

        return GenerationResult(
            prompt=prompt,
            generated_text=generated_text,
            tokens_generated=tokens_generated,
            inference_time=inference_time,
            tokens_per_second=tokens_per_second,
        )

    def evaluate_model(
        self, model, model_name: str, prompts: List[str], max_new_tokens: int = 100
    ) -> ModelMetrics:
        """
        Complete evaluation of a model.

        Args:
            model: Model to evaluate
            model_name: Name of the model
            prompts: List of prompts to test
            max_new_tokens: Tokens to generate per prompt

        Returns:
            ModelMetrics with all results
        """
        print(f"\n{'=' * 60}")
        print(f"Evaluating {model_name}")
        print(f"{'=' * 60}")

        # Evaluate losses
        train_loss, val_loss = self.evaluate_losses(model)
        train_ppl = math.exp(train_loss)
        val_ppl = math.exp(val_loss)

        print(f"Train Loss: {train_loss:.4f}, Perplexity: {train_ppl:.2f}")
        print(f"Val Loss: {val_loss:.4f}, Perplexity: {val_ppl:.2f}")

        # Generate from prompts
        print(f"\nGenerating from {len(prompts)} prompts...")
        generations = []
        total_tokens = 0
        total_time = 0.0

        for i, prompt in enumerate(prompts, 1):
            print(f"  [{i}/{len(prompts)}] Generating from: '{prompt[:50]}...'")
            result = self.generate_from_prompt(model, prompt, max_new_tokens)
            generations.append(result)
            total_tokens += result.tokens_generated
            total_time += result.inference_time
            print(
                f"      Tokens: {result.tokens_generated}, Time: {result.inference_time:.3f}s, Speed: {result.tokens_per_second:.2f} tok/s"
            )

        mean_time_per_token = total_time / total_tokens if total_tokens > 0 else 0

        print("\nSummary:")
        print(f"  Total tokens generated: {total_tokens}")
        print(f"  Total inference time: {total_time:.3f}s")
        print(f"  Mean time per token: {mean_time_per_token * 1000:.2f}ms")

        return ModelMetrics(
            model_name=model_name,
            train_loss=train_loss,
            val_loss=val_loss,
            train_perplexity=train_ppl,
            val_perplexity=val_ppl,
            total_tokens_generated=total_tokens,
            total_inference_time=total_time,
            mean_time_per_token=mean_time_per_token,
            generations=generations,
        )

    def save_results(self, gpt_metrics: ModelMetrics, lstm_metrics: ModelMetrics):
        """Save comparison results to files."""
        print(f"\n{'=' * 60}")
        print("Saving Results")
        print(f"{'=' * 60}")

        # Save GPT generations
        gpt_file = self.output_dir / "transformer_generations.txt"
        self._save_generations(gpt_metrics, gpt_file)
        print(f"Saved Transformer generations to: {gpt_file}")

        # Save LSTM generations
        lstm_file = self.output_dir / "lstm_generations.txt"
        self._save_generations(lstm_metrics, lstm_file)
        print(f"Saved LSTM generations to: {lstm_file}")

        # Save metrics comparison
        metrics_file = self.output_dir / "metrics_comparison.json"
        self._save_metrics_comparison(gpt_metrics, lstm_metrics, metrics_file)
        print(f"Saved metrics comparison to: {metrics_file}")

        # Save human-readable summary
        summary_file = self.output_dir / "comparison_summary.txt"
        self._save_summary(gpt_metrics, lstm_metrics, summary_file)
        print(f"Saved comparison summary to: {summary_file}")

        print(f"\nAll results saved to: {self.output_dir}")

    def _save_generations(self, metrics: ModelMetrics, filepath: Path):
        """Save generation results to text file."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"{'=' * 80}\n")
            f.write(f"{metrics.model_name} - Text Generations\n")
            f.write(f"{'=' * 80}\n\n")

            f.write("Model Metrics:\n")
            f.write(f"  Train Loss: {metrics.train_loss:.4f}\n")
            f.write(f"  Train Perplexity: {metrics.train_perplexity:.2f}\n")
            f.write(f"  Val Loss: {metrics.val_loss:.4f}\n")
            f.write(f"  Val Perplexity: {metrics.val_perplexity:.2f}\n")
            f.write(f"  Total Tokens Generated: {metrics.total_tokens_generated}\n")
            f.write(f"  Total Inference Time: {metrics.total_inference_time:.3f}s\n")
            f.write(
                f"  Mean Time per Token: {metrics.mean_time_per_token * 1000:.2f}ms\n"
            )
            f.write(f"\n{'=' * 80}\n\n")

            for i, gen in enumerate(metrics.generations, 1):
                f.write(f"Generation {i}/{len(metrics.generations)}\n")
                f.write(f"{'-' * 80}\n")
                f.write(f"Prompt: {gen.prompt}\n\n")
                f.write(f"Generated Text:\n{gen.generated_text}\n\n")
                f.write("Metrics:\n")
                f.write(f"  Tokens Generated: {gen.tokens_generated}\n")
                f.write(f"  Inference Time: {gen.inference_time:.3f}s\n")
                f.write(f"  Tokens per Second: {gen.tokens_per_second:.2f}\n")
                f.write(f"\n{'=' * 80}\n\n")

    def _save_metrics_comparison(
        self, gpt_metrics: ModelMetrics, lstm_metrics: ModelMetrics, filepath: Path
    ):
        """Save metrics comparison as JSON."""
        comparison = {
            "transformer": {
                "train_loss": gpt_metrics.train_loss,
                "val_loss": gpt_metrics.val_loss,
                "train_perplexity": gpt_metrics.train_perplexity,
                "val_perplexity": gpt_metrics.val_perplexity,
                "total_tokens_generated": gpt_metrics.total_tokens_generated,
                "total_inference_time": gpt_metrics.total_inference_time,
                "mean_time_per_token_ms": gpt_metrics.mean_time_per_token * 1000,
                "tokens_per_second": gpt_metrics.total_tokens_generated
                / gpt_metrics.total_inference_time
                if gpt_metrics.total_inference_time > 0
                else 0,
            },
            "lstm": {
                "train_loss": lstm_metrics.train_loss,
                "val_loss": lstm_metrics.val_loss,
                "train_perplexity": lstm_metrics.train_perplexity,
                "val_perplexity": lstm_metrics.val_perplexity,
                "total_tokens_generated": lstm_metrics.total_tokens_generated,
                "total_inference_time": lstm_metrics.total_inference_time,
                "mean_time_per_token_ms": lstm_metrics.mean_time_per_token * 1000,
                "tokens_per_second": lstm_metrics.total_tokens_generated
                / lstm_metrics.total_inference_time
                if lstm_metrics.total_inference_time > 0
                else 0,
            },
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

    def _save_summary(
        self, gpt_metrics: ModelMetrics, lstm_metrics: ModelMetrics, filepath: Path
    ):
        """Save human-readable comparison summary."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL COMPARISON SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            # Table header
            f.write(f"{'Metric':<35} {'Transformer':>20} {'LSTM':>20}\n")
            f.write("-" * 80 + "\n")

            # Loss metrics
            f.write(
                f"{'Train Loss':<35} {gpt_metrics.train_loss:>20.4f} {lstm_metrics.train_loss:>20.4f}\n"
            )
            f.write(
                f"{'Train Perplexity':<35} {gpt_metrics.train_perplexity:>20.2f} {lstm_metrics.train_perplexity:>20.2f}\n"
            )
            f.write(
                f"{'Val Loss':<35} {gpt_metrics.val_loss:>20.4f} {lstm_metrics.val_loss:>20.4f}\n"
            )
            f.write(
                f"{'Val Perplexity':<35} {gpt_metrics.val_perplexity:>20.2f} {lstm_metrics.val_perplexity:>20.2f}\n"
            )
            f.write("-" * 80 + "\n")

            # Generation metrics
            f.write(
                f"{'Total Tokens Generated':<35} {gpt_metrics.total_tokens_generated:>20} {lstm_metrics.total_tokens_generated:>20}\n"
            )
            f.write(
                f"{'Total Inference Time (s)':<35} {gpt_metrics.total_inference_time:>20.3f} {lstm_metrics.total_inference_time:>20.3f}\n"
            )
            f.write(
                f"{'Mean Time per Token (ms)':<35} {gpt_metrics.mean_time_per_token * 1000:>20.2f} {lstm_metrics.mean_time_per_token * 1000:>20.2f}\n"
            )

            gpt_tps = (
                gpt_metrics.total_tokens_generated / gpt_metrics.total_inference_time
                if gpt_metrics.total_inference_time > 0
                else 0
            )
            lstm_tps = (
                lstm_metrics.total_tokens_generated / lstm_metrics.total_inference_time
                if lstm_metrics.total_inference_time > 0
                else 0
            )
            f.write(f"{'Tokens per Second':<35} {gpt_tps:>20.2f} {lstm_tps:>20.2f}\n")

            f.write("\n" + "=" * 80 + "\n\n")

            # Winner analysis
            f.write("WINNER ANALYSIS:\n\n")

            if gpt_metrics.val_perplexity < lstm_metrics.val_perplexity:
                f.write(
                    f"✓ Transformer has LOWER perplexity ({gpt_metrics.val_perplexity:.2f} vs {lstm_metrics.val_perplexity:.2f})\n"
                )
            else:
                f.write(
                    f"✓ LSTM has LOWER perplexity ({lstm_metrics.val_perplexity:.2f} vs {gpt_metrics.val_perplexity:.2f})\n"
                )

            if gpt_tps > lstm_tps:
                f.write(
                    f"✓ Transformer is FASTER ({gpt_tps:.2f} vs {lstm_tps:.2f} tokens/s)\n"
                )
            else:
                f.write(
                    f"✓ LSTM is FASTER ({lstm_tps:.2f} vs {gpt_tps:.2f} tokens/s)\n"
                )

    def compare(self, prompts: List[str], max_new_tokens: int = 100):
        """
        Run complete comparison.

        Args:
            prompts: List of prompts to test
            max_new_tokens: Tokens to generate per prompt
        """
        # Evaluate GPT
        gpt_metrics = self.evaluate_model(
            self.gpt_model, "Transformer", prompts, max_new_tokens
        )

        # Evaluate LSTM
        lstm_metrics = self.evaluate_model(
            self.lstm_model, "LSTM", prompts, max_new_tokens
        )

        # Save results
        self.save_results(gpt_metrics, lstm_metrics)

        print(f"\n{'=' * 60}")
        print("COMPARISON COMPLETE!")
        print(f"{'=' * 60}")


def main():
    """Main comparison script."""
    # Load settings
    settings = Settings()

    # Define test prompts (10 exemplary prompts in Polish)
    prompts = [
        "Warszawa to stolica",
        "Historia Polski rozpoczęła się",
        "Największą rzeką w Polsce jest",
        "W roku 1945",
        "Polscy naukowcy odkryli",
        "Kraków jest miastem",
        "Literatura polska słynie z",
        "Podczas II wojny światowej",
        "Polska kuchnia jest znana z",
        "Jan Paweł II był",
    ]

    encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    print("Loading data...")
    with open("datasets/high_quality_plwiki.txt", "r", encoding="utf-8") as f:
        text = f.read()

    torch.manual_seed(1337)

    # Train and test splits
    data = torch.tensor(encoding.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    print(f"Train data: {len(train_data):,} tokens")
    print(f"Val data: {len(val_data):,} tokens")

    # Initialize comparator
    comparator = ModelComparator(
        settings=settings,
        train_data=train_data,
        val_data=val_data,
        gpt_checkpoint="checkpoints/gpt4.pt",
        lstm_checkpoint="checkpoints/lstm_model.pt",
        output_dir="comparison_results",
    )

    # Run comparison
    comparator.compare(prompts=prompts, max_new_tokens=100)


if __name__ == "__main__":
    main()
