# Copyright Pathway Technology, Inc.

import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

import bdh

PATH = "checkpoints/bdh.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# On a Mac you can also try
# device=torch.device('mps')

dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    torch.amp.autocast(device_type=device.type, dtype=ptdtype)
    if "cuda" in device.type
    else nullcontext()
)
scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == "float16"))
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
print(f"Using device: {device} with dtype {dtype}")


# Configuration
BDH_CONFIG = bdh.BDHConfig()
BLOCK_SIZE = 512
BATCH_SIZE = 32
MAX_ITERS = 3000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
LOG_FREQ = 100

input_file_path = os.path.join(
    os.path.dirname(__file__), "datasets/high_quality_plwiki.txt"
)


def get_batch(split):
    # FIX 1: Check if file exists
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Input file not found: {input_file_path}")

    # treat the file as bytes
    data = np.memmap(input_file_path, dtype=np.uint8, mode="r")

    # FIX 2: Handle edge case where data is too short
    if len(data) <= BLOCK_SIZE:
        raise ValueError(
            f"Data file too short ({len(data)} bytes). Need at least {BLOCK_SIZE + 1} bytes."
        )

    if split == "train":
        data = data[: int(0.9 * len(data))]
    else:
        data = data[int(0.9 * len(data)) :]

    # FIX 3: Ensure we have enough data for sampling
    if len(data) <= BLOCK_SIZE:
        raise ValueError(
            f"{split} split too short ({len(data)} bytes). Need at least {BLOCK_SIZE + 1} bytes."
        )

    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + BLOCK_SIZE]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + BLOCK_SIZE]).astype(np.int64))
            for i in ix
        ]
    )
    if torch.cuda.is_available():
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = (
            x.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True),
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def eval(model):
    model.eval()


if __name__ == "__main__":
    # FIX 4: Create checkpoints directory if it doesn't exist
    checkpoint_dir = os.path.dirname(PATH)
    if checkpoint_dir:  # Only create if path has a directory component
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    model = bdh.BDH(BDH_CONFIG).to(device)
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    x, y = get_batch("train")

    loss_acc = 0
    loss_steps = 0
    for step in range(MAX_ITERS):
        # FIX 5: Save checkpoint less frequently (every LOG_FREQ steps, not every step!)
        if step % LOG_FREQ == 0:
            torch.save(model.state_dict(), PATH)
            print(f"Checkpoint saved at step {step}")

        with ctx:
            logits, loss = model(x, y)

        # FIX 6: Get next batch AFTER computing loss (moved after loss computation)
        loss_acc += loss
        loss_steps += 1

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # Get next batch for next iteration
        x, y = get_batch("train")

        if step % LOG_FREQ == 0:
            print(
                f"Step: {step}/{MAX_ITERS} loss {loss_acc.item() / loss_steps:.3f}"
            )  # FIX 7: Added 'f' for proper formatting
            loss_acc = 0
            loss_steps = 0

    # FIX 8: Save final checkpoint
    torch.save(model.state_dict(), PATH)
    print(f"Final checkpoint saved to {PATH}")

    print("Training done, now generating samples")
    model.eval()

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

    # FIX 9: Create output file directory if needed
    output_file_path = os.path.join(os.path.dirname(__file__), "bdh_output.txt")
    output_dir = os.path.dirname(output_file_path)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # FIX 10: Open file once in append mode, not overwrite for each prompt
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("BDH Model - Generated Samples\n")
        f.write("=" * 80 + "\n\n")

        for i, prompt in enumerate(prompts, 1):
            print(f"Generating sample {i}/{len(prompts)}: {prompt}")

            prompt_tensor = torch.tensor(
                bytearray(prompt, "utf-8"), dtype=torch.long, device=device
            ).unsqueeze(0)

            # FIX 11: Add error handling for generation
            try:
                with torch.no_grad():  # FIX 12: Disable gradients during generation
                    ret = model.generate(prompt_tensor, max_new_tokens=100, top_k=3)

                ret_decoded = bytes(ret.to(torch.uint8).to("cpu").squeeze(0)).decode(
                    errors="backslashreplace"
                )

                f.write(f"Sample {i}/{len(prompts)}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Prompt: {prompt}\n\n")
                f.write(f"Generated Text:\n{ret_decoded}\n\n")
                f.write("=" * 80 + "\n\n")

            except Exception as e:
                print(f"Error generating sample {i}: {e}")
                f.write(f"Sample {i}/{len(prompts)}\n")
                f.write(f"Prompt: {prompt}\n")
                f.write(f"ERROR: {e}\n\n")
                f.write("=" * 80 + "\n\n")

    print(f"All samples saved to {output_file_path}")
