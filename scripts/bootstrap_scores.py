"""Compute paper-style stratified bootstrap statistics from sample scores."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def tpr_at_limit(labels: np.ndarray, scores: np.ndarray, limit: float) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    indices = np.flatnonzero(fpr <= limit)
    return float(tpr[indices[-1]]) if indices.size else 0.0


def describe(values: np.ndarray) -> dict:
    return {
        "mean_percent": 100.0 * float(values.mean()),
        "std_percent": 100.0 * float(values.std(ddof=1)),
        "ci95_percent": [100.0 * float(x) for x in
                         np.quantile(values, [0.025, 0.975])],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_scores", type=Path)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.sample_scores.suffix.lower() == ".csv":
        with args.sample_scores.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = json.loads(args.sample_scores.read_text())
    members = np.asarray([float(x["score"]) for x in rows
                          if int(x["label"]) == 1])
    nonmembers = np.asarray([float(x["score"]) for x in rows
                             if int(x["label"]) == 0])
    if not len(members) or not len(nonmembers):
        raise ValueError("Both classes must be present in the sample-score file.")
    labels = np.r_[np.ones(len(members), dtype=np.int8),
                   np.zeros(len(nonmembers), dtype=np.int8)]
    rng = np.random.default_rng(args.seed)
    aucs = np.empty(args.iterations)
    tpr5 = np.empty(args.iterations)
    tpr1 = np.empty(args.iterations)
    for index in range(args.iterations):
        scores = np.r_[
            members[rng.integers(0, len(members), len(members))],
            nonmembers[rng.integers(0, len(nonmembers), len(nonmembers))],
        ]
        aucs[index] = roc_auc_score(labels, scores)
        tpr5[index] = tpr_at_limit(labels, scores, 0.05)
        tpr1[index] = tpr_at_limit(labels, scores, 0.01)
    result = {
        "source": args.sample_scores.name,
        "iterations": args.iterations,
        "seed": args.seed,
        "resampling": "stratified sample-level bootstrap",
        "n_member": len(members),
        "n_nonmember": len(nonmembers),
        "auroc": describe(aucs),
        "tpr_at_5fpr": describe(tpr5),
        "tpr_at_1fpr": describe(tpr1),
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
