from pathlib import Path

from runner import ExperimentRunner
from utils import find_max_batch_size

from config import FLASH_ATTN_AVAILABLE, ExperimentConfig
from data import TextDataset, load_data
from models import GPTLanguageModel


def run_all_experiments():
    """Run comprehensive experimental comparison with real data"""

    print("\n" + "=" * 70)
    print("  COMPREHENSIVE TRANSFORMER OPTIMIZATION EXPERIMENTS")
    print("=" * 70)

    # ========================================================================
    # DATA LOADING
    # ========================================================================
    data_path = "datasets/high_quality_plwiki.txt"

    if not Path(data_path).exists():
        print(f"\nError: Data file not found at {data_path}")
        print("   Please ensure the file exists or update the path.")
        return

    # Load data
    train_data, val_data, vocab_size, encode, decode = load_data(data_path)

    # ========================================================================
    # BASE CONFIGURATION
    # ========================================================================
    base_config = ExperimentConfig(
        n_layer=6,
        n_head=8,
        n_embd=512,
        block_size=256,
        vocab_size=vocab_size,
        dropout=0.1,
        learning_rate=3e-4,
        max_iters=1000,
        eval_interval=100,
    )

    print("\nConfiguration:")
    print(f"Layers: {base_config.n_layer}")
    print(f"Heads: {base_config.n_head}")
    print(f"Embedding dim: {base_config.n_embd}")
    print(f"Block size: {base_config.block_size}")
    print(f"Vocabulary size: {base_config.vocab_size}")

    # Create datasets
    print("\nCreating PyTorch datasets...")
    train_dataset = TextDataset(train_data, base_config.block_size)
    val_dataset = TextDataset(val_data, base_config.block_size)

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")

    runner = ExperimentRunner()
    all_results = []

    # ========================================================================
    # EXPERIMENT 0: Baseline (Full Precision)
    # ========================================================================
    print("\n" + "EXPERIMENT 0: BASELINE (Full Precision)".center(70, "="))

    config_baseline = ExperimentConfig(**vars(base_config))

    # Find max batch size for baseline
    max_bs_baseline = find_max_batch_size(
        config_baseline, GPTLanguageModel, train_dataset
    )
    config_baseline.batch_size = max_bs_baseline

    results_baseline = runner.run_experiment(
        config_baseline, train_dataset, val_dataset
    )
    all_results.append(("Baseline", results_baseline))

    # ========================================================================
    # EXPERIMENT 1: BF16 Automatic Mixed Precision
    # ========================================================================
    print("\n" + "EXPERIMENT 1: BF16 AMP".center(70, "="))

    # 1a. Same batch size as baseline
    config_amp_same = ExperimentConfig(**vars(base_config))
    config_amp_same.use_amp = True
    config_amp_same.batch_size = max_bs_baseline

    print("\nPart 1a: BF16 AMP with same batch size as baseline")
    results_amp_same = runner.run_experiment(
        config_amp_same, train_dataset, val_dataset
    )
    all_results.append(("BF16-AMP (same BS)", results_amp_same))

    # 1b. Maximum batch size for AMP
    config_amp_max = ExperimentConfig(**vars(base_config))
    config_amp_max.use_amp = True

    max_bs_amp = find_max_batch_size(
        config_amp_max, GPTLanguageModel, train_dataset, start_bs=max_bs_baseline
    )
    config_amp_max.batch_size = max_bs_amp

    print("\nPart 1b: BF16 AMP with maximum batch size")
    results_amp_max = runner.run_experiment(config_amp_max, train_dataset, val_dataset)
    all_results.append(("BF16-AMP (max BS)", results_amp_max))

    # ========================================================================
    # EXPERIMENT 2: FlashAttention
    # ========================================================================
    if FLASH_ATTN_AVAILABLE:
        print("\n" + "EXPERIMENT 2: FlashAttention".center(70, "="))

        # 2a. FlashAttention with same batch size as baseline
        config_flash_same = ExperimentConfig(**vars(base_config))
        config_flash_same.use_amp = True
        config_flash_same.use_flash_attention = True
        config_flash_same.batch_size = max_bs_baseline

        print("\nPart 2a: FlashAttention with same batch size as baseline")
        results_flash_same = runner.run_experiment(
            config_flash_same, train_dataset, val_dataset
        )
        all_results.append(("FlashAttn (same BS)", results_flash_same))

        # 2b. FlashAttention with maximum batch size
        config_flash_max = ExperimentConfig(**vars(base_config))
        config_flash_max.use_amp = True
        config_flash_max.use_flash_attention = True

        max_bs_flash = find_max_batch_size(
            config_flash_max, GPTLanguageModel, train_dataset, start_bs=max_bs_amp
        )
        config_flash_max.batch_size = max_bs_flash

        print("\nPart 2b: FlashAttention with maximum batch size")
        results_flash_max = runner.run_experiment(
            config_flash_max, train_dataset, val_dataset
        )
        all_results.append(("FlashAttn (max BS)", results_flash_max))
    else:
        print("\nSkipping FlashAttention experiments (not installed)")

    # ========================================================================
    # EXPERIMENT 3: Windowed (Local) Attention
    # ========================================================================
    if FLASH_ATTN_AVAILABLE:
        print("\n" + "EXPERIMENT 3: Local Attention".center(70, "="))

        window_sizes = [128, 256]

        for window_size in window_sizes:
            print(f"\nTesting window size: {window_size}")

            # 3a. Same batch size as baseline
            config_local_same = ExperimentConfig(**vars(base_config))
            config_local_same.use_amp = True
            config_local_same.use_local_attention = True
            config_local_same.local_attention_window = window_size
            config_local_same.batch_size = max_bs_baseline

            print(f"\nPart 3a: Local Attention (W={window_size}) with same BS")
            results_local_same = runner.run_experiment(
                config_local_same, train_dataset, val_dataset
            )
            all_results.append(
                (f"LocalAttn-W{window_size} (same BS)", results_local_same)
            )

            # 3b. Maximum batch size
            config_local_max = ExperimentConfig(**vars(base_config))
            config_local_max.use_amp = True
            config_local_max.use_local_attention = True
            config_local_max.local_attention_window = window_size

            max_bs_local = find_max_batch_size(
                config_local_max, GPTLanguageModel, train_dataset, start_bs=max_bs_amp
            )
            config_local_max.batch_size = max_bs_local

            print(f"\n   Part 3b: Local Attention (W={window_size}) with max BS")
            results_local_max = runner.run_experiment(
                config_local_max, train_dataset, val_dataset
            )
            all_results.append(
                (f"LocalAttn-W{window_size} (max BS)", results_local_max)
            )
    else:
        print("\nSkipping Local Attention experiments (flash-attn not installed)")

    # ========================================================================
    # EXPERIMENT 4: Gradient Checkpointing
    # ========================================================================
    print("\n" + "EXPERIMENT 4: Gradient Checkpointing".center(70, "="))

    # 4a. With baseline settings
    config_gc_same = ExperimentConfig(**vars(base_config))
    config_gc_same.use_amp = True
    config_gc_same.use_gradient_checkpointing = True
    config_gc_same.batch_size = max_bs_baseline

    print("\nPart 4a: Gradient Checkpointing with same BS")
    results_gc_same = runner.run_experiment(config_gc_same, train_dataset, val_dataset)
    all_results.append(("GradCP (same BS)", results_gc_same))

    # 4b. Maximum batch size with gradient checkpointing
    config_gc_max = ExperimentConfig(**vars(base_config))
    config_gc_max.use_amp = True
    config_gc_max.use_gradient_checkpointing = True

    max_bs_gc = find_max_batch_size(
        config_gc_max, GPTLanguageModel, train_dataset, start_bs=max_bs_amp
    )
    config_gc_max.batch_size = max_bs_gc

    print("\n Part 4b: Gradient Checkpointing with max BS")
    results_gc_max = runner.run_experiment(config_gc_max, train_dataset, val_dataset)
    all_results.append(("GradCP (max BS)", results_gc_max))

    # ========================================================================
    # COMBINED: FlashAttention + Gradient Checkpointing
    # ========================================================================
    if FLASH_ATTN_AVAILABLE:
        print("\n" + "EXPERIMENT 5: FlashAttention + GradCP".center(70, "="))

        config_combined = ExperimentConfig(**vars(base_config))
        config_combined.use_amp = True
        config_combined.use_flash_attention = True
        config_combined.use_gradient_checkpointing = True

        max_bs_combined = find_max_batch_size(
            config_combined, GPTLanguageModel, train_dataset, start_bs=max_bs_gc
        )
        config_combined.batch_size = max_bs_combined

        results_combined = runner.run_experiment(
            config_combined, train_dataset, val_dataset
        )
        all_results.append(("FlashAttn+GradCP (max BS)", results_combined))

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY".center(70))
    print("=" * 70)

    print(
        f"\n{'Experiment':<35} {'BS':<8} {'Perplexity':<12} {'Time(s)':<10} {'Peak Mem(MB)':<12}"
    )
    print("-" * 77)

    for name, results in all_results:
        bs = results["config"].split("BS")[1].split("_")[0]
        ppl = results["final_perplexity"]
        time_s = results["total_time_seconds"]
        mem = results["final_memory_stats"].get("peak_allocated_mb", 0)

        print(f"{name:<35} {bs:<8} {ppl:<12.2f} {time_s:<10.1f} {mem:<12.0f}")

    # Calculate improvements over baseline
    baseline_ppl = all_results[0][1]["final_perplexity"]
    baseline_time = all_results[0][1]["total_time_seconds"]
    baseline_mem = all_results[0][1]["final_memory_stats"].get("peak_allocated_mb", 0)

    print("\n" + "=" * 70)
    print("IMPROVEMENTS OVER BASELINE".center(70))
    print("=" * 70)
    print(
        f"\n{'Experiment':<35} {'PPL Change':<15} {'Speedup':<12} {'Mem Reduction':<15}"
    )
    print("-" * 77)

    for name, results in all_results[1:]:  # Skip baseline
        ppl = results["final_perplexity"]
        time_s = results["total_time_seconds"]
        mem = results["final_memory_stats"].get("peak_allocated_mb", 0)

        ppl_change = ((ppl - baseline_ppl) / baseline_ppl) * 100
        speedup = baseline_time / time_s
        mem_reduction = ((baseline_mem - mem) / baseline_mem) * 100

        print(
            f"{name:<35} {ppl_change:>+6.2f}%         "
            f"{speedup:>5.2f}x       {mem_reduction:>+6.2f}%"
        )

    print("\n" + "=" * 70)
    print("All experiments completed successfully!")
    print("=" * 70)

    # Save summary
    summary_path = Path("experiment_results/summary.txt")
    summary_path.parent.mkdir(exist_ok=True)

    with open(summary_path, "w") as f:
        f.write("EXPERIMENT SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        for name, results in all_results:
            f.write(f"{name}\n")
            f.write(f"  Perplexity: {results['final_perplexity']:.2f}\n")
            f.write(f"  Time: {results['total_time_seconds']:.1f}s\n")
            f.write(
                f"  Peak Memory: {results['final_memory_stats'].get('peak_allocated_mb', 0):.0f} MB\n\n"
            )

    print(f"Summary saved to: {summary_path}")

    return all_results


if __name__ == "__main__":
    run_all_experiments()
