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
"""Score regex, model and combined detection on the held-out test split.

A predicted span counts as a true positive when its label matches a gold span and the two overlap
with Jaccard >= 0.5 (character level); each gold span matches at most once.

    ./evaluate.py --data data --model model --out metrics.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

from pii import LABELS, Span, Tagger, combine, regex_spans, valid

JACCARD = 0.5


def jaccard(a: Span, g: dict) -> float:
    inter = max(0, min(a.end, g["end"]) - max(a.start, g["start"]))
    union = max(a.end, g["end"]) - min(a.start, g["start"])
    return inter / union if union else 0.0


def score(rows: list[dict], preds: list[list[Span]]) -> dict:
    counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in LABELS}
    for row, spans in zip(rows, preds):
        used: set[int] = set()
        for p in spans:
            hit = next(
                (
                    i
                    for i, g in enumerate(row["spans"])
                    if i not in used and g["label"] == p.label and jaccard(p, g) >= JACCARD
                ),
                None,
            )
            if hit is None:
                counts[p.label]["fp"] += 1
            else:
                used.add(hit)
                counts[p.label]["tp"] += 1
        for i, g in enumerate(row["spans"]):
            if i not in used:
                counts[g["label"]]["fn"] += 1

    def prf(c: dict) -> dict:
        p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else None
        r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else None
        f = 2 * p * r / (p + r) if p and r else 0.0
        return {**c, "precision": p, "recall": r, "f1": f}

    total = {k: sum(c[k] for c in counts.values()) for k in ("tp", "fp", "fn")}
    return {"overall": prf(total), "by_label": {label: prf(c) for label, c in counts.items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=pathlib.Path, default=pathlib.Path("data"))
    parser.add_argument("--model", default="model")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("metrics.json"))
    args = parser.parse_args()

    rows = [json.loads(line) for line in (args.data / "test.jsonl").open()]
    texts = [r["text"] for r in rows]
    tagger = Tagger(args.model)

    t0 = time.perf_counter()
    model = []
    for i in range(0, len(texts), 64):
        model += tagger.spans(texts[i : i + 64])
    model_s = time.perf_counter() - t0
    regex = [regex_spans(t) for t in texts]
    combined = [combine(t, m, r) for t, m, r in zip(texts, model, regex)]

    # Concrete wins for the page, spread across entity types: model-only catches of free-text
    # entities, and regex hits that a validator rejects.
    wins: dict[str, list[dict]] = {}
    saves: dict[str, list[dict]] = {}
    for row, m, r in zip(rows, model, regex):
        regex_labels = {(s.start, s.label) for s in r}
        for s in m:
            correct = any(g["label"] == s.label and jaccard(s, g) >= JACCARD for g in row["spans"])
            if correct and s.label in ("NAME", "ADDRESS") and (s.start, s.label) not in regex_labels:
                wins.setdefault(s.label, []).append({"text": row["text"], "span": s.as_dict(row["text"])})
        for s in r:
            gold = any(g["label"] == s.label and jaccard(s, g) >= JACCARD for g in row["spans"])
            if not gold and not valid(s.label, row["text"], s.start, s.end):
                saves.setdefault(s.label, []).append({"text": row["text"], "span": s.as_dict(row["text"])})

    def spread(by_label: dict[str, list[dict]], k: int) -> list[dict]:
        out: list[dict] = []
        for i in range(k):
            for label in sorted(by_label):
                if i < len(by_label[label]) and len(out) < k:
                    out.append(by_label[label][i])
        return out

    model_wins, validator_saves = spread(wins, 6), spread(saves, 6)

    result = {
        "test_lines": len(rows),
        "gold_spans": sum(len(r["spans"]) for r in rows),
        "match_rule": f"same label, character Jaccard >= {JACCARD}",
        "split": "held-out templates and name/street/city pools",
        "modes": {"regex": score(rows, regex), "model": score(rows, model), "combined": score(rows, combined)},
        "local_model_lines_per_s": round(len(rows) / model_s, 1),
        "examples": {"model_wins": model_wins, "validator_saves": validator_saves},
    }
    args.out.write_text(json.dumps(result, indent=2))
    for mode, s in result["modes"].items():
        o = s["overall"]
        print(
            f"{mode:9s} P={o['precision']:.3f} R={o['recall']:.3f} F1={o['f1']:.3f} "
            f"(tp={o['tp']} fp={o['fp']} fn={o['fn']})"
        )


if __name__ == "__main__":
    main()
