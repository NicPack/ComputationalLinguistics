"""
Prompt generation strategies for LLM evaluation.

Design decisions:
- Template-based approach: Consistent formatting across tasks
- Model-aware routing: deepseek-r1 uses internal reasoning, no CoT injection
- Strategy as pure functions: No side effects, easy to test
- Metadata tracking: Return both prompt and config for reproducibility
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

from tasks import Task


class PromptStrategy(Enum):
    """Available prompting strategies."""

    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "cot"


@dataclass(frozen=True)
class PromptConfig:
    """
    Prompt generation configuration.

    Why frozen: Ensures prompts are reproducible - same config always
    generates same prompt.
    """

    strategy: PromptStrategy
    n_examples: int = 2  # For few-shot
    temperature: float = 0.1  # Model sampling temperature
    max_tokens: int = 2048

    def __post_init__(self):
        """Validate configuration."""
        if self.strategy == PromptStrategy.FEW_SHOT and self.n_examples < 1:
            raise ValueError("Few-shot requires n_examples >= 1")


@dataclass
class GeneratedPrompt:
    """
    Complete prompt with metadata.

    Why separate class: Bundles prompt with generation config for
    full reproducibility tracking.
    """

    text: str
    config: PromptConfig
    task_id: str
    model: str
    strategy_used: PromptStrategy  # Actual strategy (may differ from requested for reasoning models)


# =============================================================================
# CORE PROMPT STRATEGIES
# =============================================================================


def zero_shot(task: Task) -> str:
    """
    Generate zero-shot prompt: task description only.

    Args:
        task: Task to generate prompt for

    Returns:
        Formatted prompt string

    Design rationale: Minimal prompt tests model's base capabilities
    without guidance. Establishes baseline performance.
    """
    prompt = f"""{task.description}

{task.eval_example["input"]}"""

    return prompt.strip()


def few_shot(task: Task, n_examples: int = 2) -> str:
    """
    Generate few-shot prompt: examples + task.

    Args:
        task: Task to generate prompt for
        n_examples: Number of dev examples to include (default: 2)

    Returns:
        Formatted prompt string

    Design rationale: Examples demonstrate expected format and quality.
    Uses dev examples (never eval example) to test generalization.

    Critical: Takes first n_examples from dev set. Ensure dev examples
    are high-quality and representative.
    """
    if n_examples > len(task.dev_examples):
        raise ValueError(
            f"Task {task.id} has only {len(task.dev_examples)} dev examples, "
            f"but {n_examples} requested"
        )

    # Build examples section
    examples_text = "Here are some examples:\n\n"

    for i, example in enumerate(task.dev_examples[:n_examples], 1):
        examples_text += f"Example {i}:\n"
        examples_text += f"{example['input']}\n\n"
        examples_text += f"Response:\n{example['output']}\n\n"
        examples_text += "---\n\n"

    # Add actual task
    prompt = f"""{task.description}

{examples_text}Now solve this:

{task.eval_example["input"]}"""

    return prompt.strip()


def chain_of_thought(task: Task) -> str:
    """
    Generate CoT prompt: task + explicit reasoning instruction.

    Args:
        task: Task to generate prompt for

    Returns:
        Formatted prompt string

    Design rationale: Explicit step-by-step trigger improves reasoning
    for standard models. Research shows "Let's think step by step"
    activates better reasoning pathways.

    CRITICAL: Do NOT use with reasoning models (deepseek-r1, qwen with
    reasoning mode) - they have internal CoT mechanisms.
    """
    cot_trigger = "Let's approach this step by step:"

    prompt = f"""{task.description}

{task.eval_example["input"]}

{cot_trigger}"""

    return prompt.strip()


# =============================================================================
# MODEL-AWARE ROUTING
# =============================================================================


def get_prompt(
    task: Task, strategy: PromptStrategy, model: str, n_examples: int = 2
) -> GeneratedPrompt:
    """
    Generate prompt with model-specific adaptations.

    Args:
        task: Task to generate prompt for
        strategy: Requested prompting strategy
        model: Model identifier (e.g., 'ministral-3:3b', 'deepseek-r1:7b')
        n_examples: Number of examples for few-shot (default: 2)

    Returns:
        GeneratedPrompt with text and metadata

    Design rationale: Central routing ensures model-specific rules are
    consistently applied. Prevents mistakes like using CoT with reasoning models.

    Model-specific rules:
    - deepseek-r1: Has internal reasoning, ignore CoT requests
    - ministral-3:3b: Standard model, supports all strategies
    - Future models: Add rules here
    """
    # Determine actual strategy to use
    actual_strategy = strategy
    reasoning_models = [
        "deepseek-r1",
        "deepseek-r1:7b",
        "qwen3",
    ]  # Models with built-in reasoning

    is_reasoning_model = any(rm in model.lower() for rm in reasoning_models)

    if is_reasoning_model and strategy == PromptStrategy.CHAIN_OF_THOUGHT:
        # Override CoT for reasoning models
        actual_strategy = PromptStrategy.ZERO_SHOT
        print(f"Warning: {model} has internal reasoning. Converting CoT → Zero-shot.")

    # Generate prompt based on actual strategy
    if actual_strategy == PromptStrategy.ZERO_SHOT:
        prompt_text = zero_shot(task)

    elif actual_strategy == PromptStrategy.FEW_SHOT:
        prompt_text = few_shot(task, n_examples)

    elif actual_strategy == PromptStrategy.CHAIN_OF_THOUGHT:
        prompt_text = chain_of_thought(task)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Determine temperature based on task
    # Creative tasks benefit from higher temperature
    creative_tasks = ["task_03_creative"]
    temperature = 0.7 if task.id in creative_tasks else 0.1

    # Build config
    config = PromptConfig(
        strategy=actual_strategy,
        n_examples=n_examples,
        temperature=temperature,
        max_tokens=2048,
    )

    return GeneratedPrompt(
        text=prompt_text,
        config=config,
        task_id=task.id,
        model=model,
        strategy_used=actual_strategy,
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_all_strategies_for_model(model: str) -> List[PromptStrategy]:
    """
    Get applicable strategies for a model.

    Args:
        model: Model identifier

    Returns:
        List of strategies that should be tested with this model

    Use case: Experiment planning - determine which combinations to run.
    """
    reasoning_models = ["deepseek-r1", "deepseek-r1:7b", "qwen3"]
    is_reasoning_model = any(rm in model.lower() for rm in reasoning_models)

    if is_reasoning_model:
        # Reasoning models: only zero-shot and few-shot
        return [PromptStrategy.ZERO_SHOT, PromptStrategy.FEW_SHOT]
    else:
        # Standard models: all strategies
        return [
            PromptStrategy.ZERO_SHOT,
            PromptStrategy.FEW_SHOT,
            PromptStrategy.CHAIN_OF_THOUGHT,
        ]


def estimate_prompt_length(prompt: str) -> int:
    """
    Rough token count estimate (1 token ≈ 4 chars).

    Args:
        prompt: Prompt text

    Returns:
        Estimated token count

    Use case: Verify prompts fit within context windows.
    Not precise, but sufficient for sanity checking.
    """
    return len(prompt) // 4


def preview_prompts(task: Task, model: str) -> Dict[str, str]:
    """
    Generate all applicable prompts for a task/model combination.

    Args:
        task: Task to generate prompts for
        model: Model identifier

    Returns:
        Dict mapping strategy names to prompt texts

    Use case: Manual inspection during prompt engineering phase.
    Helps verify prompts look reasonable before full execution.
    """
    strategies = get_all_strategies_for_model(model)
    prompts = {}

    for strategy in strategies:
        generated = get_prompt(task, strategy, model)
        prompts[strategy.value] = generated.text

    return prompts


# =============================================================================
# VALIDATION & TESTING
# =============================================================================

if __name__ == "__main__":
    from tasks import TASK_REGISTRY

    print("=== PROMPT GENERATION TESTS ===\n")

    # Test with one task and both models
    task = TASK_REGISTRY["task_02_logic"]
    models = ["ministral-3:3b", "deepseek-r1:7b"]

    for model in models:
        print(f"\n{'=' * 60}")
        print(f"Model: {model}")
        print(f"{'=' * 60}\n")

        strategies = get_all_strategies_for_model(model)
        print(f"Applicable strategies: {[s.value for s in strategies]}\n")

        for strategy in strategies:
            print(f"\n--- {strategy.value.upper()} ---")

            generated = get_prompt(task, strategy, model, n_examples=2)

            print(f"Strategy used: {generated.strategy_used.value}")
            print(f"Temperature: {generated.config.temperature}")
            print(f"Estimated tokens: {estimate_prompt_length(generated.text)}")
            print("\nPrompt preview (first 300 chars):")
            print(generated.text[:300] + "...\n")

    # Test CoT override for reasoning model
    print("\n" + "=" * 60)
    print("Testing CoT override for reasoning model")
    print("=" * 60 + "\n")

    generated = get_prompt(task, PromptStrategy.CHAIN_OF_THOUGHT, "deepseek-r1:7b")

    print(f"Requested: {PromptStrategy.CHAIN_OF_THOUGHT.value}")
    print(f"Actually used: {generated.strategy_used.value}")
    print("✓ Correctly converted CoT → Zero-shot for reasoning model")

    # Test few-shot with different example counts
    print("\n" + "=" * 60)
    print("Testing few-shot with different example counts")
    print("=" * 60 + "\n")

    for n in [1, 2, 3]:
        generated = get_prompt(
            task, PromptStrategy.FEW_SHOT, "ministral-3:3b", n_examples=n
        )
        print(f"n_examples={n}: {estimate_prompt_length(generated.text)} tokens")

    print("\n✓ All tests passed")
