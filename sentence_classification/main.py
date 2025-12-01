from pathlib import Path

import torch
from torch.utils.data import DataLoader
from train import fine_tune_pretrained, train_from_scratch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

from datasets import load_dataset
from evaluate import compare_experiments, inference_benchmark

if __name__ == "__main__":
    # Setup
    device = "cuda" if torch.cuda.is_available() else "mps"
    print(f"Using device: {device}")

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("jziebura/polish_youth_slang_classification")
    train_data = dataset["train"]
    val_data = dataset["validation"]

    num_classes = len(set(train_data["sentyment"]))
    print(f"Number of classes: {num_classes}")

    # ========================================================================
    # EXPERIMENT 1: TRAIN FROM SCRATCH
    # ========================================================================
    print("\n" + "=" * 80)
    print("PREPARING FROM-SCRATCH EXPERIMENT")
    print("=" * 80)

    from_scratch_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="my_tokenizers/bielik_tokenizer/tokenizer.json",
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
    )

    print(f"From-scratch tokenizer vocab size: {from_scratch_tokenizer.vocab_size}")

    # Tokenize for from-scratch model
    def scratch_tokenize_batch(examples):
        tokenized = from_scratch_tokenizer(
            examples["tekst"], truncation=True, padding="max_length", max_length=128
        )
        # Rename label column to 'label' for consistency
        tokenized["label"] = examples["sentyment"]
        return tokenized

    print("Tokenizing data for from-scratch model...")
    scratch_train_data = train_data.map(scratch_tokenize_batch, batched=True)
    scratch_val_data = val_data.map(scratch_tokenize_batch, batched=True)

    scratch_train_data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"],
    )
    scratch_val_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )

    # Validate token IDs
    all_token_ids = []
    for seq in scratch_train_data["input_ids"]:
        all_token_ids.extend(seq)
    max_token_id = max(all_token_ids)
    min_token_id = min(all_token_ids)

    print(f"Max token ID in data: {max_token_id}")
    print(f"Vocab size: {from_scratch_tokenizer.vocab_size}")
    assert max_token_id < from_scratch_tokenizer.vocab_size, "Token IDs out of range!"

    # Create dataloaders
    scratch_train_dataloader = DataLoader(
        scratch_train_data, batch_size=32, shuffle=True
    )
    scratch_val_dataloader = DataLoader(scratch_val_data, batch_size=64)

    # Model config for from-scratch
    scratch_model_config = {
        "n_head": 4,
        "block_size": 128,
        "n_embd": 128,
        "n_layer": 3,
        "vocab_size": from_scratch_tokenizer.vocab_size,
        "dropout": 0.5,
        "device": device,
        "num_labels": num_classes,
    }

    scratch_training_config = {
        "num_epochs": 40,
        "learning_rate": 1e-5,
        "weight_decay": 0.1,
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0,
        "save_every": 5,
    }

    # Train from scratch
    print("\nTraining from scratch...")
    scratch_model, scratch_metrics, scratch_timing = train_from_scratch(
        train_dataloader=scratch_train_dataloader,
        val_dataloader=scratch_val_dataloader,
        model_config=scratch_model_config,
        training_config=scratch_training_config,
        save_dir=Path("results/from_scratch"),
        device=device,
    )

    # ========================================================================
    # EXPERIMENT 2: FINE-TUNE PRE-TRAINED MODEL
    # ========================================================================
    print("\n" + "=" * 80)
    print("PREPARING FINE-TUNING EXPERIMENT")
    print("=" * 80)

    pretrained_model_name = "speakleash/Bielik-1.5B-v3"

    # Load pre-trained tokenizer
    pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    print(f"Pre-trained tokenizer vocab size: {pretrained_tokenizer.vocab_size}")

    # Load pre-trained model
    print("Loading pre-trained Bielik model...")
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name,
        torch_dtype=torch.float32,  # Use float32 for CPU/MPS
        low_cpu_mem_usage=True,
    )

    # Tokenize for pre-trained model
    def pretrained_tokenize_batch(examples):
        tokenized = pretrained_tokenizer(
            examples["tekst"], truncation=True, padding="max_length", max_length=128
        )
        tokenized["label"] = examples["sentyment"]
        return tokenized

    print("Tokenizing data for pre-trained model...")
    pretrained_train_data = train_data.map(pretrained_tokenize_batch, batched=True)
    pretrained_val_data = val_data.map(pretrained_tokenize_batch, batched=True)

    pretrained_train_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
    pretrained_val_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )

    # Validate token IDs
    all_token_ids = []
    for seq in scratch_train_data["input_ids"]:
        all_token_ids.extend(seq)
    max_token_id = max(all_token_ids)
    min_token_id = min(all_token_ids)
    print(f"Max token ID in data: {max_token_id}")
    print(f"Vocab size: {pretrained_tokenizer.vocab_size}")
    assert max_token_id < pretrained_tokenizer.vocab_size, "Token IDs out of range!"

    # Create dataloaders
    pretrained_train_dataloader = DataLoader(
        pretrained_train_data, batch_size=32, shuffle=True
    )
    pretrained_val_dataloader = DataLoader(pretrained_val_data, batch_size=64)

    # CRITICAL: Use pre-trained model's actual config
    pretrained_model_config = {
        "n_head": pretrained_model.config.num_attention_heads,
        "block_size": 128,  # Your sequence length
        "n_embd": pretrained_model.config.hidden_size,
        "n_layer": pretrained_model.config.num_hidden_layers,
        "vocab_size": pretrained_tokenizer.vocab_size,
        "dropout": 0.4,
        "device": device,
        "num_labels": num_classes,
    }

    print("\nPre-trained model config:")
    print(f"  Vocab size: {pretrained_model_config['vocab_size']}")
    print(f"  Hidden size: {pretrained_model_config['n_embd']}")
    print(f"  Num layers: {pretrained_model_config['n_layer']}")
    print(f"  Num heads: {pretrained_model_config['n_head']}")

    finetune_training_config = {
        "num_epochs": 5,
        "learning_rate": 2e-6,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0,
        "save_every": 1,
    }

    # Save pre-trained model in expected format for fine_tune_pretrained
    pretrained_checkpoint_path = Path("results/pretrained_bielik.pt")
    pretrained_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving pre-trained model to: {pretrained_checkpoint_path}")
    torch.save(
        {
            "model_state_dict": pretrained_model.state_dict(),
            "config": pretrained_model.config,
        },
        pretrained_checkpoint_path,
    )

    # Fine-tune
    print("\nFine-tuning pre-trained model...")
    finetuned_model, finetuned_metrics, finetuned_timing = fine_tune_pretrained(
        train_dataloader=pretrained_train_dataloader,
        val_dataloader=pretrained_val_dataloader,
        pretrained_model_path=pretrained_checkpoint_path,
        model_config=pretrained_model_config,
        training_config=finetune_training_config,
        save_dir=Path("results/fine_tuned"),
        device=device,
        freeze_backbone=False,
    )

    # ========================================================================
    # EVALUATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("BENCHMARKING INFERENCE")
    print("=" * 80)

    scratch_inference = inference_benchmark(
        scratch_model, scratch_val_dataloader, device
    )

    finetuned_inference = inference_benchmark(
        finetuned_model,
        pretrained_val_dataloader,
        device,
    )

    # Compare results
    print("\n" + "=" * 80)
    print("COMPARING RESULTS")
    print("=" * 80)

    compare_experiments(
        from_scratch_dir=Path("results/from_scratch"),
        fine_tuned_dir=Path("results/fine_tuned"),
        save_dir=Path("results/comparison"),
    )

    print("\n✅ All experiments completed!")
