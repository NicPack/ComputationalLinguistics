import json
from pathlib import Path

import torch
from tokenizer_utils import load_tokenizer

import bdh
from config import CONFIG


def analyze_tokenization(text_samples, tokenizers):
    """
    Analyze how different tokenizers handle the same text.

    Returns detailed comparison of tokenization strategies.
    """
    results = []

    for idx, text in enumerate(text_samples, 1):
        sample_analysis = {
            "sample_id": idx,
            "text": text,
            "word_count": len(text.split()),
            "char_count": len(text),
            "tokenizers": {},
        }

        for tok_type, tokenizer in tokenizers.items():
            tokens = tokenizer.tokenize(text)
            token_ids = tokenizer.encode(text)

            # Count words that are single tokens
            words = text.split()
            words_as_single_token = 0
            for word in words:
                word_tokens = tokenizer.tokenize(word)
                if len(word_tokens) == 1:
                    words_as_single_token += 1

            sample_analysis["tokenizers"][tok_type] = {
                "tokens": tokens,
                "token_ids": token_ids[:20],  # First 20 for readability
                "token_count": len(tokens),
                "tokens_per_word": len(tokens) / len(words) if words else 0,
                "words_as_single_token": words_as_single_token,
                "pct_words_as_single_token": (
                    words_as_single_token / len(words) * 100 if words else 0
                ),
            }

        results.append(sample_analysis)

    return results


def generate_samples(tokenizer_type, model, tokenizer, prompts, device):
    """Generate text samples from trained model."""
    model.eval()

    samples = []

    for prompt in prompts:
        try:
            # Encode prompt
            prompt_ids = tokenizer.encode(prompt)
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

            # Generate
            with torch.no_grad():
                output_ids = model.generate(
                    prompt_tensor,
                    max_new_tokens=CONFIG.generation_max_tokens,
                    top_k=CONFIG.generation_top_k,
                )

            # Decode
            output_text = tokenizer.decode(output_ids[0].tolist())

            samples.append(
                {
                    "prompt": prompt,
                    "generated": output_text,
                    "success": True,
                }
            )

        except Exception as e:
            samples.append(
                {
                    "prompt": prompt,
                    "generated": None,
                    "success": False,
                    "error": str(e),
                }
            )

    return samples


def main():
    """
    Run comprehensive qualitative analysis across all tokenizers.
    """
    print(f"\n{'=' * 80}")
    print("Qualitative Analysis: Tokenizer Comparison")
    print(f"{'=' * 80}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load all tokenizers
    print("Loading tokenizers...")
    tokenizers = {
        "bielik": load_tokenizer("bielik", CONFIG),
        "whitespace": load_tokenizer("whitespace", CONFIG),
        "sentencepiece": load_tokenizer("sentencepiece", CONFIG),
    }

    # === TOKENIZATION ANALYSIS ===
    print(f"\n{'=' * 80}")
    print("Part 1: Tokenization Analysis")
    print(f"{'=' * 80}\n")

    # Sample texts for analysis (Polish text, 30+ words each)
    analysis_samples = [
        "Warszawa jest stolicą Polski i największym miastem w kraju. "
        "Miasto leży nad Wisłą, w centralnej części Polski, w województwie mazowieckim. "
        "Warszawa jest ważnym ośrodkiem kulturalnym, naukowym i gospodarczym. "
        "Historia miasta sięga średniowiecza, a jego rozwój przyspieszył w czasach renesansu.",
        "Polscy naukowcy wnieśli znaczący wkład w rozwój nauki światowej. "
        "Maria Skłodowska-Curie była pierwszą kobietą, która otrzymała Nagrodę Nobla. "
        "Jej odkrycia w dziedzinie radioaktywności zmieniły oblicze fizyki i chemii. "
        "Polska szkoła matematyczna zasłynęła również na arenie międzynarodowej.",
        "Literatura polska słynie z bogatej tradycji poetyckiej i prozatorskiej. "
        "Adam Mickiewicz, uznawany za największego polskiego poetę romantyzmu, "
        "stworzył arcydzieła takie jak 'Pan Tadeusz'. Współczesna literatura polska "
        "kontynuuje tę tradycję, zdobywając międzynarodowe uznanie i nagrody literackie.",
    ]

    print("Analyzing tokenization of sample texts...\n")
    tokenization_results = analyze_tokenization(analysis_samples, tokenizers)

    # Print human-readable comparison
    for result in tokenization_results:
        print(f"\nSample {result['sample_id']}:")
        print(f"Text: {result['text'][:100]}...")
        print(f"Words: {result['word_count']}, Characters: {result['char_count']}")
        print()

        for tok_type in ["bielik", "whitespace", "sentencepiece"]:
            tok_data = result["tokenizers"][tok_type]
            print(f"  {tok_type.upper()}:")
            print(f"    Tokens: {tok_data['token_count']}")
            print(f"    Tokens/word: {tok_data['tokens_per_word']:.3f}")
            print(
                f"    Words as single token: {tok_data['words_as_single_token']} "
                f"({tok_data['pct_words_as_single_token']:.1f}%)"
            )
            print(f"    First 10 tokens: {tok_data['tokens'][:10]}")
            print()

    # === GENERATION COMPARISON ===
    print(f"\n{'=' * 80}")
    print("Part 2: Generated Samples Comparison")
    print(f"{'=' * 80}\n")

    generation_results = {}

    for tok_type in ["bielik", "whitespace", "sentencepiece"]:
        print(f"\nGenerating samples with {tok_type} model...")

        # Load model
        checkpoint_path = Path(CONFIG.checkpoint_dir) / f"bdh_{tok_type}.pt"

        if not checkpoint_path.exists():
            print(f"  Checkpoint not found: {checkpoint_path}")
            print(f"  Skipping {tok_type}")
            continue

        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model = bdh.BDH(checkpoint["config"]).to(device)

        # Handle torch.compile wrapper prefix
        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        model.eval()

        # Generate
        samples = generate_samples(
            tok_type,
            model,
            tokenizers[tok_type],
            CONFIG.generation_prompts[:3],  # Use first 3 prompts
            device,
        )

        generation_results[tok_type] = samples

        # Print samples
        for sample in samples:
            print(f"\n  Prompt: {sample['prompt']}")
            if sample["success"]:
                print(f"  Generated: {sample['generated'][:200]}...")
            else:
                print(f"  ERROR: {sample['error']}")

    # === SAVE RESULTS ===
    print(f"\n{'=' * 80}")
    print("Saving Results")
    print(f"{'=' * 80}\n")

    results = {
        "tokenization_analysis": tokenization_results,
        "generation_samples": generation_results,
    }

    output_path = Path(CONFIG.results_dir) / "qualitative_analysis.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_path}")

    # === CREATE HUMAN-READABLE REPORT ===
    report_path = Path(CONFIG.results_dir) / "qualitative_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("QUALITATIVE ANALYSIS REPORT\n")
        f.write("Tokenizer Comparison for Polish Language Model\n")
        f.write("=" * 80 + "\n\n")

        f.write("PART 1: TOKENIZATION ANALYSIS\n")
        f.write("-" * 80 + "\n\n")

        for result in tokenization_results:
            f.write(f"\nSample {result['sample_id']}\n")
            f.write(f"{'=' * 80}\n")
            f.write(
                f"Text ({result['word_count']} words, {result['char_count']} chars):\n"
            )
            f.write(f"{result['text']}\n\n")

            # Create comparison table
            f.write(
                f"{'Tokenizer':<15} {'Tokens':<10} {'Tok/Word':<12} {'Single-Tok Words':<20}\n"
            )
            f.write(f"{'-' * 15} {'-' * 10} {'-' * 12} {'-' * 20}\n")

            for tok_type in ["bielik", "whitespace", "sentencepiece"]:
                tok_data = result["tokenizers"][tok_type]
                f.write(
                    f"{tok_type:<15} "
                    f"{tok_data['token_count']:<10} "
                    f"{tok_data['tokens_per_word']:<12.3f} "
                    f"{tok_data['words_as_single_token']} ({tok_data['pct_words_as_single_token']:.1f}%)\n"
                )

            f.write("\nTokenization Examples:\n")
            for tok_type in ["bielik", "whitespace", "sentencepiece"]:
                tok_data = result["tokenizers"][tok_type]
                f.write(f"\n{tok_type.upper()}:\n")
                f.write(f"  {tok_data['tokens'][:15]}\n")

            f.write("\n" + "=" * 80 + "\n")

        f.write("\n\nPART 2: GENERATED SAMPLES\n")
        f.write("-" * 80 + "\n\n")

        for tok_type in ["bielik", "whitespace", "sentencepiece"]:
            if tok_type not in generation_results:
                continue

            f.write(f"\n{tok_type.upper()} MODEL\n")
            f.write("=" * 80 + "\n")

            for sample in generation_results[tok_type]:
                f.write(f"\nPrompt: {sample['prompt']}\n")
                f.write("-" * 80 + "\n")
                if sample["success"]:
                    f.write(f"{sample['generated']}\n")
                else:
                    f.write(f"ERROR: {sample['error']}\n")
                f.write("\n")

    print(f"Human-readable report saved to: {report_path}")

    print(f"\n{'=' * 80}")
    print("Qualitative analysis complete!")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
