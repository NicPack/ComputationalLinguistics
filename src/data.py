import os
from pathlib import Path
from typing import Generator

import torch
from speakleash import Speakleash


def load_speakleash_data(data_path: str, dataset_name: str):
    sl = Speakleash(data_path)
    ds = sl.get(dataset_name).ext_data

    return ds


def extract_quality_files(dataloader: Generator, dir: str, desired_quality: str):
    # Iterate over documents in the dataset
    counter = 0
    for index, doc in enumerate(dataloader):
        txt, meta = doc
        doc_quality: str = meta.get("quality", "").lower()
        if desired_quality is None:
            desired_quality = doc_quality
        if doc_quality == desired_quality:
            counter += 1
            with open(
                os.path.join(dir, f"{desired_quality}_quality_doc_{counter}.txt"),
                "w",
                encoding="utf-8",
            ) as out_file:
                out_file.write(txt)


def get_quality_files_from_dataset(
    data_path: str, dataset_name: str, quality: str | None = None
):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    replicate_to = os.path.join(base_dir, data_path)

    dir_to_extract_files_to = Path(os.path.join(replicate_to, dataset_name))
    dir_to_extract_files_to.mkdir(parents=True, exist_ok=True)

    ds = load_speakleash_data(replicate_to, dataset_name)
    extract_quality_files(ds, dir_to_extract_files_to, quality)


def write_speakleash_dataset_into_single_file(
    dataset_name: str, data_dir: str, quality: str | None = None
):
    get_quality_files_from_dataset(data_dir, dataset_name, quality)

    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(base_dir, data_dir)
    txt_files = next(os.walk(f"{data_path}/{dataset_name}"))[2]

    for txt_file in txt_files:
        with open(f"{data_dir}/{dataset_name}/{txt_file}", "r") as in_file:
            quality_str = quality if quality else "all"
            with open(
                f"{data_dir}/{quality_str}_quality_{dataset_name}.txt", "a"
            ) as out_file:
                for line in in_file:
                    out_file.write(line)


def get_batch(block_size: int, batch_size: int, data: torch.tensor, device):
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(
    model, eval_iters, block_size, batch_size, train_data, val_data, device
):
    out = {}
    mapping = {"train": train_data, "val": val_data}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(
                block_size=block_size,
                batch_size=batch_size,
                data=mapping[split],
                device=device,
            )
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
