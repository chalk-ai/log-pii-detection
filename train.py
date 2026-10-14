#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["torch==2.8.0", "transformers==4.56.2"]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Fine-tune distilbert-base-cased (Apache-2.0) to tag PII in log lines.

./synth.py --out data
./train.py --data data --out model        # about 33 minutes on 48 CPU cores for 2 epochs
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import time

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from pii import TAGS

BASE = "distilbert/distilbert-base-cased"
MAX_LEN = 160


def load(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def encode(tokenizer, rows: list[dict]) -> dict[str, torch.Tensor]:
    enc = tokenizer(
        [r["text"] for r in rows],
        return_offsets_mapping=True,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_tensors="pt",
    )
    labels = torch.full(enc["input_ids"].shape, -100, dtype=torch.long)
    for i, row in enumerate(rows):
        for j, (s, e) in enumerate(enc["offset_mapping"][i].tolist()):
            if s == e:
                continue
            tag = "O"
            for span in row["spans"]:
                if s >= span["start"] and e <= span["end"]:
                    tag = ("B-" if s == span["start"] else "I-") + span["label"]
                    break
            labels[i, j] = TAGS.index(tag)
    enc.pop("offset_mapping")
    enc["labels"] = labels
    return enc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=pathlib.Path, default=pathlib.Path("data"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("model"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    random.seed(0)

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForTokenClassification.from_pretrained(
        BASE, num_labels=len(TAGS), id2label=dict(enumerate(TAGS)), label2id={t: i for i, t in enumerate(TAGS)}
    )
    train = encode(tokenizer, load(args.data / "train.jsonl"))
    n = train["input_ids"].shape[0]
    steps = args.epochs * (n // args.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: max(0.0, 1 - step / steps))

    model.train()
    t0, step = time.time(), 0
    for epoch in range(args.epochs):
        order = torch.randperm(n)
        for b in range(n // args.batch):
            idx = order[b * args.batch : (b + 1) * args.batch]
            loss = model(**{k: v[idx] for k, v in train.items()}).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    (args.out / "training.json").write_text(
        json.dumps(
            {
                "base_model": BASE,
                "base_license": "apache-2.0",
                "train_lines": n,
                "epochs": args.epochs,
                "seconds": round(time.time() - t0, 1),
                "device": "cpu",
            },
            indent=2,
        )
    )
    print(f"saved {args.out} after {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
