"""
Experiment orchestration and execution.

Design decisions:
- Dataclass configs: Immutable, hashable experiment specifications
- JSON output: Human-readable, easily parseable results
- Resumable execution: Skip already-completed experiments
- Dry-run mode: Validate setup before expensive inference
- Progress tracking: Clear logging for long-running experiments
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from prompts import PromptStrategy, get_all_strategies_for_model, get_prompt
from tasks import TASK_REGISTRY, Task

from models import OllamaClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Complete specification of a single experiment.

    Why frozen: Enables hashing for deduplication and cache lookups.
    Immutable configs prevent accidental modification during execution.
    """

    task_id: str
    model: str
    strategy: PromptStrategy
    n_examples: int = 2  # Relevant for few-shot
    run_id: str = ""  # Unique identifier for this run (e.g., timestamp)

    def to_filename(self) -> str:
        """
        Generate unique, readable filename for this experiment.

        Returns:
            Sanitized filename like: task_01_instruction__ministral-3_3b__few_shot.json

        Design rationale: Filenames are self-documenting and sortable.
        Easy to find specific experiments without opening files.
        """
        # Sanitize model name (replace : and / with _)
        safe_model = self.model.replace(":", "_").replace("/", "_")

        return f"{self.task_id}__{safe_model}__{self.strategy.value}.json"

    def __str__(self) -> str:
        """Human-readable representation."""
        return f"{self.task_id} | {self.model} | {self.strategy.value}"


@dataclass
class ExperimentResult:
    """
    Complete result from a single experiment execution.

    Not frozen: Mutable to allow adding scoring results later.
    """

    config: ExperimentConfig
    prompt: str
    response: str
    model_metadata: dict[str, Any]  # tokens, duration, cached, etc.
    timestamp: str
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to JSON-serializable dict.

        Returns:
            dict with all fields, handling nested dataclasses

        Design rationale: Explicit serialization logic prevents surprises
        with nested objects and enums.
        """
        return {
            "config": {
                "task_id": self.config.task_id,
                "model": self.config.model,
                "strategy": self.config.strategy.value,
                "n_examples": self.config.n_examples,
                "run_id": self.config.run_id,
            },
            "prompt": self.prompt,
            "response": self.response,
            "model_metadata": self.model_metadata,
            "timestamp": self.timestamp,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentResult":
        """Reconstruct from saved JSON."""
        config = ExperimentConfig(
            task_id=data["config"]["task_id"],
            model=data["config"]["model"],
            strategy=PromptStrategy(data["config"]["strategy"]),
            n_examples=data["config"]["n_examples"],
            run_id=data["config"].get("run_id", ""),
        )

        return cls(
            config=config,
            prompt=data["prompt"],
            response=data["response"],
            model_metadata=data["model_metadata"],
            timestamp=data["timestamp"],
            error=data.get("error"),
        )


class ExperimentRunner:
    """
    Orchestrates LLM evaluation experiments.

    Responsibilities:
    - Generate experiment matrix (all task/model/strategy combinations)
    - Execute experiments with proper error handling
    - Save results with full metadata
    - Support resumable execution (skip completed)
    - Provide progress feedback
    """

    def __init__(
        self,
        client: OllamaClient,
        tasks: dict[str, Task],
        models: List[str],
        output_dir: str = "results/raw_outputs",
    ):
        """
        Args:
            client: Configured OllamaClient instance
            tasks: Task registry (typically TASK_REGISTRY)
            models: List of model identifiers to evaluate
            output_dir: Where to save experiment results
        """
        self.client = client
        self.tasks = tasks
        self.models = models
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique run ID based on timestamp
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info(f"Initialized ExperimentRunner with run_id: {self.run_id}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Models: {models}")
        logger.info(f"Tasks: {len(tasks)}")

    def generate_experiment_matrix(self) -> List[ExperimentConfig]:
        """
        Generate all experiment combinations.

        Returns:
            List of ExperimentConfig for each task/model/strategy combo

        Design rationale: Pre-compute all experiments for:
        - Total count estimation
        - Resume capability (check which are done)
        - Dry-run validation
        - Progress tracking

        For your setup:
        - 10 tasks
        - 2 models (ministral-3:3b with 3 strategies, deepseek-r1:7b with 2)
        - = 10 * (3 + 2) = 50 experiments
        """
        experiments = []

        for task_id, task in self.tasks.items():
            for model in self.models:
                # Get applicable strategies for this model
                strategies = get_all_strategies_for_model(model)

                for strategy in strategies:
                    config = ExperimentConfig(
                        task_id=task_id,
                        model=model,
                        strategy=strategy,
                        n_examples=2,  # Standard for all few-shot experiments
                        run_id=self.run_id,
                    )
                    experiments.append(config)

        logger.info(f"Generated {len(experiments)} experiment configurations")
        return experiments

    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Execute a single experiment.

        Args:
            config: Experiment specification

        Returns:
            ExperimentResult with response and metadata

        Design rationale: Single responsibility - run one experiment.
        Error handling ensures failures don't crash entire run.
        All context saved for debugging.
        """
        logger.info(f"Running: {config}")

        try:
            # Get task
            task = self.tasks[config.task_id]

            # Generate prompt
            generated_prompt = get_prompt(
                task=task,
                strategy=config.strategy,
                model=config.model,
                n_examples=config.n_examples,
            )

            # Query model
            model_response = self.client.generate(
                prompt=generated_prompt.text,
                model=config.model,
                temperature=generated_prompt.config.temperature,
                max_tokens=generated_prompt.config.max_tokens,
            )

            if model_response:
                # Build result
                result = ExperimentResult(
                    config=config,
                    prompt=generated_prompt.text,
                    response=model_response["response"],
                    model_metadata={
                        "prompt_tokens": model_response["prompt_tokens"],
                        "response_tokens": model_response["response_tokens"],
                        "total_duration": model_response["total_duration"],
                        "cached": model_response["cached"],
                        "temperature": generated_prompt.config.temperature,
                        "max_tokens": generated_prompt.config.max_tokens,
                    },
                    timestamp=datetime.now().isoformat(),
                    error=None,
                )

                logger.info(
                    f"✓ Completed in {model_response['total_duration']:.2f}s "
                    f"({model_response['response_tokens']} tokens)"
                )

                return result

        except Exception as e:
            error = e
            logger.error(f"✗ Failed: {e}")

        # Return error result for debugging
        return ExperimentResult(
            config=config,
            prompt="[Error: prompt generation failed]",
            response="[Error: see error field]",
            model_metadata={},
            timestamp=datetime.now().isoformat(),
            error=str(error),
        )

    def save_result(self, result: ExperimentResult) -> Path:
        """
        Save experiment result to JSON.

        Args:
            result: Experiment result to save

        Returns:
            Path where result was saved

        Design rationale: Individual files per experiment enable:
        - Incremental saving (don't lose all work on crash)
        - Easy resume (check which files exist)
        - Parallel processing (future enhancement)
        """
        filepath = self.output_dir / result.config.to_filename()

        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        return filepath

    def load_completed_configs(self) -> set:
        """
        Find already-completed experiments.

        Returns:
            Set of ExperimentConfig for completed experiments

        Use case: Resume interrupted runs without re-running expensive inference.
        """
        completed = set()

        for filepath in self.output_dir.glob("*.json"):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    result = ExperimentResult.from_dict(data)

                    # Only count as completed if no error
                    if result.error is None:
                        completed.add(result.config)
            except Exception as e:
                logger.warning(f"Could not load {filepath}: {e}")

        return completed

    def run_all(
        self, dry_run: bool = False, force_rerun: bool = False
    ) -> List[ExperimentResult]:
        """
        Execute all experiments in the matrix.

        Args:
            dry_run: If True, only print what would be run (no execution)
            force_rerun: If True, re-run even if results exist

        Returns:
            List of all experiment results

        Design rationale:
        - Dry-run validates setup before committing time
        - Resume capability saves time on interrupted runs
        - Progress logging helps estimate completion time
        - Results returned for immediate analysis if desired
        """
        experiments = self.generate_experiment_matrix()

        if dry_run:
            logger.info("=== DRY RUN MODE ===")
            logger.info(f"Would execute {len(experiments)} experiments:")

            # Group by model for clarity
            by_model = {}
            for exp in experiments:
                by_model.setdefault(exp.model, []).append(exp)

            for model, exps in by_model.items():
                logger.info(f"\n{model}: {len(exps)} experiments")
                strategies = {}
                for exp in exps:
                    strategies.setdefault(exp.strategy.value, 0)
                    strategies[exp.strategy.value] += 1

                for strategy, count in strategies.items():
                    logger.info(f"  - {strategy}: {count}")

            logger.info(f"\nOutput directory: {self.output_dir}")
            logger.info("Run with dry_run=False to execute.")
            return []

        # Check for completed experiments (unless force_rerun)
        if not force_rerun:
            completed = self.load_completed_configs()
            experiments = [e for e in experiments if e not in completed]

            if completed:
                logger.info(
                    f"Found {len(completed)} completed experiments, skipping..."
                )

        if not experiments:
            logger.info(
                "All experiments already completed! Use force_rerun=True to re-run."
            )
            return []

        # Execute experiments
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Starting experiment run: {self.run_id}")
        logger.info(f"Total experiments to run: {len(experiments)}")
        logger.info(f"{'=' * 60}\n")

        results = []

        for i, config in enumerate(experiments, 1):
            logger.info(f"\n[{i}/{len(experiments)}] {config}")

            # Run experiment
            result = self.run_experiment(config)

            # Save immediately (don't lose work on crash)
            filepath = self.save_result(result)
            logger.info(f"Saved to: {filepath.name}")

            results.append(result)

        # Summary
        logger.info(f"\n{'=' * 60}")
        logger.info("EXPERIMENT RUN COMPLETE")
        logger.info(f"{'=' * 60}")
        logger.info(f"Total experiments: {len(results)}")

        successful = sum(1 for r in results if r.error is None)
        failed = len(results) - successful

        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")

        if failed > 0:
            logger.warning("\nFailed experiments:")
            for result in results:
                if result.error:
                    logger.warning(f"  - {result.config}: {result.error}")

        logger.info(f"\nResults saved to: {self.output_dir}")

        return results

    def get_summary_stats(self) -> dict[str, Any]:
        """
        Generate summary statistics for completed experiments.

        Returns:
            dict with counts, token usage, timing stats

        Use case: Quick overview of experimental results before scoring.
        """
        completed = self.load_completed_configs()

        if not completed:
            return {"message": "No completed experiments found"}

        # Load all results
        results = []
        for config in completed:
            filepath = self.output_dir / config.to_filename()
            with open(filepath, "r") as f:
                data = json.load(f)
                results.append(ExperimentResult.from_dict(data))

        # Compute stats
        total_tokens = sum(r.model_metadata["response_tokens"] for r in results)
        total_duration = sum(r.model_metadata["total_duration"] for r in results)
        cached_count = sum(1 for r in results if r.model_metadata["cached"])

        # Group by model
        by_model = {}
        for result in results:
            model = result.config.model
            by_model.setdefault(model, []).append(result)

        model_stats = {}
        for model, model_results in by_model.items():
            model_stats[model] = {
                "count": len(model_results),
                "strategies": list(set(r.config.strategy.value for r in model_results)),
                "total_tokens": sum(
                    r.model_metadata["response_tokens"] for r in model_results
                ),
                "avg_duration": sum(
                    r.model_metadata["total_duration"] for r in model_results
                )
                / len(model_results),
            }

        return {
            "total_experiments": len(results),
            "total_tokens_generated": total_tokens,
            "total_duration_seconds": total_duration,
            "cached_responses": cached_count,
            "cache_hit_rate": cached_count / len(results) if results else 0,
            "by_model": model_stats,
        }


# =============================================================================
# CLI INTERFACE
# =============================================================================


def main():
    """
    Command-line interface for running experiments.

    Usage:
        python -m src.runner                    # Dry-run
        python -m src.runner --execute          # Full run
        python -m src.runner --execute --force  # Force re-run all
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM evaluation experiments")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run experiments (default: dry-run only)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-run even if results exist"
    )
    parser.add_argument(
        "--output",
        default="results/raw_outputs",
        help="Output directory (default: results/raw_outputs)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ministral-3:3b", "deepseek-r1:7b"],
        help="Models to evaluate (default: ministral-3:3b deepseek-r1:7b)",
    )

    args = parser.parse_args()

    # Initialize components
    client = OllamaClient()

    runner = ExperimentRunner(
        client=client, tasks=TASK_REGISTRY, models=args.models, output_dir=args.output
    )

    # Run experiments
    if args.execute:
        results = runner.run_all(dry_run=False, force_rerun=args.force)

        # Print summary stats
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        stats = runner.get_summary_stats()
        print(json.dumps(stats, indent=2))

    else:
        # Dry-run by default
        runner.run_all(dry_run=True)
        print("\nTo execute, run with --execute flag")


if __name__ == "__main__":
    main()
