"""
Manual evaluation and scoring of experiment results.

Design decisions:
- Interactive CLI: Clear display, easy input, progress tracking
- Incremental saving: Each score saved immediately (no lost work)
- Resumable sessions: Skip already-scored results
- Structured scores: Separate file per result for easy analysis
- Validation: Enforce 1-5 scale, required fields
"""

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from runner import ExperimentResult
from tasks import TASK_REGISTRY, EvaluationCriterion, Task

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class CriterionScore:
    """Score for a single evaluation criterion."""

    criterion_name: str
    score: int  # 1-5
    justification: str

    def __post_init__(self):
        """Validate score range."""
        if not 1 <= self.score <= 5:
            raise ValueError(f"Score must be 1-5, got {self.score}")


@dataclass
class Score:
    """
    Complete evaluation score for one experiment.

    Structure mirrors ExperimentResult for easy joining during analysis.
    """

    experiment_id: str  # Unique identifier (matches result filename)
    task_id: str
    model: str
    strategy: str
    criterion_scores: List[CriterionScore]
    overall_notes: str
    scorer: str  # Who performed the evaluation
    timestamp: str

    @property
    def mean_score(self) -> float:
        """Average score across all criteria."""
        if not self.criterion_scores:
            return 0.0
        return sum(cs.score for cs in self.criterion_scores) / len(
            self.criterion_scores
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "experiment_id": self.experiment_id,
            "task_id": self.task_id,
            "model": self.model,
            "strategy": self.strategy,
            "criterion_scores": [
                {
                    "criterion_name": cs.criterion_name,
                    "score": cs.score,
                    "justification": cs.justification,
                }
                for cs in self.criterion_scores
            ],
            "mean_score": self.mean_score,
            "overall_notes": self.overall_notes,
            "scorer": self.scorer,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Score":
        """Reconstruct from saved JSON."""
        criterion_scores = [
            CriterionScore(
                criterion_name=cs["criterion_name"],
                score=cs["score"],
                justification=cs["justification"],
            )
            for cs in data["criterion_scores"]
        ]

        return cls(
            experiment_id=data["experiment_id"],
            task_id=data["task_id"],
            model=data["model"],
            strategy=data["strategy"],
            criterion_scores=criterion_scores,
            overall_notes=data["overall_notes"],
            scorer=data["scorer"],
            timestamp=data["timestamp"],
        )


class EvaluationSession:
    """
    Interactive scoring session for experiment results.

    Responsibilities:
    - Load experiment results
    - Display results with context (task, rubric)
    - Collect scores via CLI
    - Validate input
    - Save scores incrementally
    - Track progress
    """

    def __init__(
        self,
        results_dir: str = "llm_eval/results/raw_outputs",
        tasks: Dict[str, Task] = TASK_REGISTRY,
        scores_dir: str = "llm_eval/results/scores",
        scorer_name: str = "human",
    ):
        """
        Args:
            results_dir: Directory containing experiment results
            scores_dir: Directory to save scores
            tasks: Task registry (default: TASK_REGISTRY)
            scorer_name: Identifier for who's scoring (useful for multi-annotator studies)
        """
        self.results_dir = Path(results_dir)
        self.scores_dir = Path(scores_dir)
        self.scores_dir.mkdir(parents=True, exist_ok=True)

        self.tasks = tasks
        self.scorer_name = scorer_name

        logger.info("Initialized EvaluationSession")
        logger.info(f"Results dir: {self.results_dir}")
        logger.info(f"Scores dir: {self.scores_dir}")
        logger.info(f"Scorer: {self.scorer_name}")

    def load_results(self) -> List[ExperimentResult]:
        """
        Load all experiment results from results directory.

        Returns:
            List of ExperimentResult objects

        Design rationale: Load all at once for progress tracking.
        Could be optimized to stream for very large result sets.
        """
        results = []

        for filepath in sorted(self.results_dir.glob("*.json")):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    result = ExperimentResult.from_dict(data)

                    # Skip failed experiments
                    if result.error is not None:
                        logger.warning(f"Skipping failed experiment: {filepath.name}")
                        continue

                    results.append(result)

            except Exception as e:
                logger.error(f"Failed to load {filepath}: {e}")

        logger.info(f"Loaded {len(results)} experiment results")
        return results

    def load_completed_scores(self) -> set:
        """
        Find already-scored experiments.

        Returns:
            Set of experiment IDs (filenames) that have scores

        Use case: Resume interrupted scoring sessions.
        """
        completed = set()

        for filepath in self.scores_dir.glob("*.json"):
            completed.add(filepath.stem)  # Filename without extension

        if completed:
            logger.info(f"Found {len(completed)} already-scored experiments")

        return completed

    def _display_header(self, result: ExperimentResult, index: int, total: int):
        """Display experiment context."""
        print("\n" + "=" * 80)
        print(f"EXPERIMENT [{index}/{total}]")
        print("=" * 80)
        print(f"Task: {result.config.task_id}")
        print(f"Model: {result.config.model}")
        print(f"Strategy: {result.config.strategy.value}")
        print(f"Timestamp: {result.timestamp}")
        print("=" * 80)

    def _display_prompt(self, prompt: str):
        """Display prompt with formatting."""
        print("\n" + "-" * 80)
        print("PROMPT:")
        print("-" * 80)
        print(prompt)
        print("-" * 80)

    def _display_response(self, response: str):
        """Display model response with formatting."""
        print("\n" + "-" * 80)
        print("MODEL RESPONSE:")
        print("-" * 80)
        print(response)
        print("-" * 80)

    def _display_rubric(self, criteria: List[EvaluationCriterion]):
        """Display evaluation criteria."""
        print("\n" + "-" * 80)
        print("EVALUATION CRITERIA:")
        print("-" * 80)
        for i, criterion in enumerate(criteria, 1):
            print(f"\n{i}. {criterion.name}")
            print(f"   Description: {criterion.description}")
            print(f"   Scale: {criterion.scale}")
        print("-" * 80)

    def _get_score_input(self, criterion: EvaluationCriterion) -> CriterionScore:
        """
        Interactive input for a single criterion score.

        Args:
            criterion: Criterion to score

        Returns:
            CriterionScore with validated input

        Design rationale: Separate function for testability and reuse.
        Clear prompts, validation, error messages.
        """
        print(f"\n--- Scoring: {criterion.name} ---")
        print(f"Scale: {criterion.scale}")

        # Get score (with validation)
        while True:
            try:
                score_input = input("Score (1-5): ").strip()
                score = int(score_input)

                if 1 <= score <= 5:
                    break
                else:
                    print("❌ Score must be between 1 and 5. Try again.")
            except ValueError:
                print("❌ Please enter a number between 1 and 5.")
            except KeyboardInterrupt:
                print("\n\nScoring interrupted by user.")
                sys.exit(0)

        # Get justification
        print("Justification (press Enter for none):")
        try:
            justification = input("> ").strip()
        except KeyboardInterrupt:
            print("\n\nScoring interrupted by user.")
            sys.exit(0)

        return CriterionScore(
            criterion_name=criterion.name,
            score=score,
            justification=justification or "[No justification provided]",
        )

    def score_result(self, result: ExperimentResult, index: int, total: int) -> Score:
        """
        Interactive scoring for a single experiment result.

        Args:
            result: Experiment result to score
            index: Current position (for progress display)
            total: Total number of results

        Returns:
            Complete Score object

        Design rationale: All display and input logic in one place.
        Clear structure: header → prompt → response → rubric → scoring.
        """
        # Get task for rubric
        task = self.tasks[result.config.task_id]

        # Display context
        self._display_header(result, index, total)
        self._display_prompt(result.prompt)
        self._display_response(result.response)
        self._display_rubric(task.criteria)

        # Collect scores for each criterion
        print("\n" + "=" * 80)
        print("SCORING")
        print("=" * 80)

        criterion_scores = []
        for criterion in task.criteria:
            score = self._get_score_input(criterion)
            criterion_scores.append(score)

        # Overall notes
        print("\n" + "-" * 80)
        print("Overall notes/observations (press Enter for none):")
        try:
            overall_notes = input("> ").strip()
        except KeyboardInterrupt:
            print("\n\nScoring interrupted by user.")
            sys.exit(0)

        # Build Score object
        score = Score(
            experiment_id=result.config.to_filename().replace(".json", ""),
            task_id=result.config.task_id,
            model=result.config.model,
            strategy=result.config.strategy.value,
            criterion_scores=criterion_scores,
            overall_notes=overall_notes or "[No overall notes]",
            scorer=self.scorer_name,
            timestamp=datetime.now().isoformat(),
        )

        # Display summary
        print("\n" + "-" * 80)
        print("SCORE SUMMARY:")
        for cs in criterion_scores:
            print(f"  {cs.criterion_name}: {cs.score}/5")
        print(f"  Mean: {score.mean_score:.2f}/5")
        print("-" * 80)

        return score

    def save_score(self, score: Score) -> Path:
        """
        Save score to JSON file.

        Args:
            score: Score to save

        Returns:
            Path where score was saved

        Design rationale: Same filename as result for easy joining.
        Individual files enable incremental saving.
        """
        filepath = self.scores_dir / f"{score.experiment_id}.json"

        with open(filepath, "w") as f:
            json.dump(score.to_dict(), f, indent=2)

        logger.info(f"Saved score to: {filepath.name}")
        return filepath

    def batch_evaluate(
        self, skip_completed: bool = True, limit: Optional[int] = None
    ) -> List[Score]:
        """
        Score all experiment results interactively.

        Args:
            skip_completed: If True, skip already-scored experiments
            limit: Optional limit on number to score (for testing/partial sessions)

        Returns:
            List of all scores collected in this session

        Design rationale: Main entry point for scoring workflow.
        Progress tracking, resume capability, graceful interruption handling.
        """
        # Load results
        results = self.load_results()

        if not results:
            logger.warning("No results found to score!")
            return []

        # Filter completed if requested
        if skip_completed:
            completed = self.load_completed_scores()
            results = [
                r
                for r in results
                if r.config.to_filename().replace(".json", "") not in completed
            ]

        if not results:
            logger.info(
                "All results already scored! Use skip_completed=False to re-score."
            )
            return []

        # Apply limit if specified
        if limit:
            results = results[:limit]

        # Session header
        print("\n" + "=" * 80)
        print("EVALUATION SESSION")
        print("=" * 80)
        print(f"Total results to score: {len(results)}")
        print(f"Scorer: {self.scorer_name}")
        print(f"Scores will be saved to: {self.scores_dir}")
        print("\nPress Ctrl+C at any time to stop (progress is saved incrementally)")
        print("=" * 80)

        input("\nPress Enter to begin scoring...")

        # Score each result
        scores = []

        try:
            for i, result in enumerate(results, 1):
                # Score
                score = self.score_result(result, i, len(results))

                # Save immediately
                self.save_score(score)

                scores.append(score)

                # Brief pause between scores
                if i < len(results):
                    print("\n✓ Score saved. Ready for next experiment...")
                    input("Press Enter to continue (or Ctrl+C to stop)...")

        except KeyboardInterrupt:
            print("\n\n" + "=" * 80)
            print("SESSION INTERRUPTED")
            print("=" * 80)
            print(f"Scored {len(scores)} experiments before interruption.")
            print("Progress saved. Run again to continue from where you left off.")
            print("=" * 80)

        # Final summary
        if scores:
            print("\n" + "=" * 80)
            print("SESSION COMPLETE")
            print("=" * 80)
            print(f"Total scored in this session: {len(scores)}")
            print(
                f"Mean score across all: {sum(s.mean_score for s in scores) / len(scores):.2f}/5"
            )
            print(f"Scores saved to: {self.scores_dir}")
            print("=" * 80)

        return scores

    def get_scoring_progress(self) -> Dict[str, Any]:
        """
        Get statistics on scoring progress.

        Returns:
            Dict with total, completed, remaining counts

        Use case: Check progress without starting a session.
        """
        total_results = len(self.load_results())
        completed_scores = len(self.load_completed_scores())
        remaining = total_results - completed_scores

        return {
            "total_results": total_results,
            "completed_scores": completed_scores,
            "remaining": remaining,
            "progress_percent": (completed_scores / total_results * 100)
            if total_results > 0
            else 0,
        }

    def export_scores_csv(
        self, output_path: str = "llm_eval/results/scores.csv"
    ) -> Path | None:
        """
        Export all scores to CSV for easy analysis.

        Args:
            output_path: Where to save CSV

        Returns:
            Path to saved CSV

        Design rationale: CSV is universal format for analysis tools.
        Flattens nested structure for easy import into pandas/Excel.
        """
        import csv

        # Load all scores
        scores: list[Score] = []
        for filepath in self.scores_dir.glob("*.json"):
            with open(filepath, "r") as f:
                data = json.load(f)
                scores.append(Score.from_dict(data))

        if not scores:
            logger.warning("No scores found to export")
            return None

        # Determine all unique criterion names
        all_criteria = set()
        for score in scores:
            for cs in score.criterion_scores:
                all_criteria.add(cs.criterion_name)

        all_criteria = sorted(all_criteria)

        # Write CSV
        csv_path = Path(output_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with open(csv_path, "w", newline="") as f:
            fieldnames = [
                "experiment_id",
                "task_id",
                "model",
                "strategy",
                "mean_score",
                "overall_notes",
                "scorer",
                "timestamp",
            ] + [f"score_{crit}" for crit in all_criteria]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for score in scores:
                row = {
                    "experiment_id": score.experiment_id,
                    "task_id": score.task_id,
                    "model": score.model,
                    "strategy": score.strategy,
                    "mean_score": score.mean_score,
                    "overall_notes": score.overall_notes,
                    "scorer": score.scorer,
                    "timestamp": score.timestamp,
                }

                # Add criterion scores
                for cs in score.criterion_scores:
                    row[f"score_{cs.criterion_name}"] = cs.score

                writer.writerow(row)

        logger.info(f"Exported {len(scores)} scores to {csv_path}")
        return csv_path


# =============================================================================
# CLI INTERFACE
# =============================================================================


def main():
    """
    Command-line interface for scoring.

    Usage:
        python -m src.evaluation                     # Check progress
        python -m src.evaluation --score             # Start scoring session
        python -m src.evaluation --score --limit 5   # Score 5 experiments (testing)
        python -m src.evaluation --export            # Export to CSV
    """
    import argparse

    parser = argparse.ArgumentParser(description="Score experiment results")
    parser.add_argument(
        "--score", action="store_true", help="Start interactive scoring session"
    )
    parser.add_argument(
        "--limit", type=int, help="Limit number of experiments to score (for testing)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-score even if scores exist"
    )
    parser.add_argument("--export", action="store_true", help="Export scores to CSV")
    parser.add_argument(
        "--results",
        default="llm_eval/results/raw_outputs",
        help="Results directory (default: llm_eval/results/raw_outputs)",
    )
    parser.add_argument(
        "--scores",
        default="llm_eval/results/scores",
        help="Scores directory (default: llm_eval/results/scores)",
    )
    parser.add_argument(
        "--scorer", default="human", help="Scorer identifier (default: human)"
    )

    args = parser.parse_args()

    # Initialize session
    session = EvaluationSession(
        results_dir=args.results, scores_dir=args.scores, scorer_name=args.scorer
    )

    if args.export:
        # Export to CSV
        csv_path = session.export_scores_csv()
        if csv_path:
            print(f"\n✓ Scores exported to: {csv_path}")

    elif args.score:
        # Start scoring
        scores = session.batch_evaluate(skip_completed=not args.force, limit=args.limit)

    else:
        # Show progress by default
        progress = session.get_scoring_progress()

        print("\n" + "=" * 80)
        print("SCORING PROGRESS")
        print("=" * 80)
        print(f"Total experiments: {progress['total_results']}")
        print(f"Completed scores: {progress['completed_scores']}")
        print(f"Remaining: {progress['remaining']}")
        print(f"Progress: {progress['progress_percent']:.1f}%")
        print("=" * 80)

        if progress["remaining"] > 0:
            print("\nTo start scoring, run: python -m src.evaluation --score")
        else:
            print("\n✓ All experiments scored!")
            print("To export to CSV, run: python -m src.evaluation --export")


if __name__ == "__main__":
    main()
