#!/usr/bin/env python3
"""Reproducible GradAudit for SmolVLM2 and MiniCPM-V-2.

All large gradient tensors stay on CUDA.  Only final scalar scores are copied
to host memory. Failed samples are excluded (never replaced by zero) and are
recorded explicitly. Outputs contain everything needed to recompute metrics.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
PROJECT = HERE / "new_vlm"
MODEL_ROOT = Path(os.environ.get("VLM_MIA_MODEL_ROOT", PACKAGE_ROOT / "models"))
DATA_ROOT = Path(os.environ.get(
    "VLM_MIA_DATA_ROOT", PACKAGE_ROOT / "data/minigpt_coco_gptcap_mia"))
RESULT_ROOT = Path(os.environ.get("VLM_MIA_OUTPUT_ROOT", PACKAGE_ROOT / "results"))
PORTABLE = PROJECT
sys.path.insert(0, str(PORTABLE))
sys.path.insert(0, str(PORTABLE / "adapters"))

from adapters.gen_vlm_victim import GenVLMVictim, KINDS  # noqa: E402
from gradaudit.core.data import MIARecord, make_split_labeled  # noqa: E402
from gradaudit.core.sims import row_col_cos, build_masks  # noqa: E402

MODEL_SPECS = {
    "smolvlm2": {
        "model_dir": MODEL_ROOT / "SmolVLM2-2.2B-Instruct",
        "text_pattern": r"model\.text_model\.layers\.(\d+)\.",
        "vision_pattern": r"model\.vision_model\.encoder\.layers\.(\d+)\.",
        "projector_prefixes": ("model.connector.",),
    },
    "minicpmv2": {
        "model_dir": MODEL_ROOT / "MiniCPM-V-2",
        "text_pattern": r"llm\.model\.layers\.(\d+)\.",
        # Actual MiniCPM-V-2 weights use vpm.blocks.N, not encoder.layers.N.
        "vision_pattern": r"vpm\.blocks\.(\d+)\.",
        "projector_prefixes": ("resampler.",),
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=MODEL_SPECS)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap", type=int, default=10_000)
    p.add_argument("--tau", type=float, default=0.10)
    p.add_argument("--mask-strategy", default="tau_or_topq",
                   choices=("fixed_tau", "tau_or_topq", "tau_or_topq_union",
                            "adaptive_keep"))
    p.add_argument("--topq", type=float, default=0.80)
    p.add_argument("--last-n", type=int, default=3)
    p.add_argument("--limit", type=int, default=0,
                   help="Debug only: limit each probe class; reference remains 200/class")
    return p.parse_args()


def load_split(seed: int, limit: int):
    raw = json.loads((DATA_ROOT / "dataset.json").read_text(encoding="utf-8"))
    rows = []
    for r in raw:
        cap = r.get("caption", "")
        if isinstance(cap, list):
            cap = cap[0] if cap else ""
        rows.append(MIARecord(
            image=str(DATA_ROOT / r["image"]), target=str(cap),
            label=int(r["label"]), prompt="Describe the image in detail.",
            uid=str(r.get("coco_id", r["image"])),
        ))
    split = make_split_labeled(rows, seed=seed)
    if limit:
        split.probe_members = split.probe_members[:limit]
        split.probe_nonmembers = split.probe_nonmembers[:limit]
    return split


def select_last_layers(model, spec, last_n: int):
    import re
    named = list(model.named_parameters())
    ti, vi = set(), set()
    for name, _ in named:
        mt = re.search(spec["text_pattern"], name)
        mv = re.search(spec["vision_pattern"], name)
        if mt:
            ti.add(int(mt.group(1)))
        if mv:
            vi.add(int(mv.group(1)))
    if not ti or not vi:
        raise RuntimeError(f"Layer discovery failed: text={sorted(ti)}, vision={sorted(vi)}")
    keep_t = set(sorted(ti)[-last_n:])
    keep_v = set(sorted(vi)[-last_n:])
    selected = []
    for name, param in named:
        if param.ndim < 2:
            continue
        mt = re.search(spec["text_pattern"], name)
        mv = re.search(spec["vision_pattern"], name)
        is_projector = any(name.startswith(x) for x in spec["projector_prefixes"])
        if ((mt and int(mt.group(1)) in keep_t)
                or (mv and int(mv.group(1)) in keep_v) or is_projector):
            selected.append(name)
    if not selected:
        raise RuntimeError("No trainable parameters selected")
    return selected, sorted(keep_t), sorted(keep_v)


def tpr_at(labels, scores, target):
    fpr, tpr, _ = roc_curve(labels, scores)
    ok = np.flatnonzero(fpr <= target + 1e-12)
    return float(tpr[ok].max()) if len(ok) else 0.0


def metrics(labels, scores):
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "tpr_at_1fpr": tpr_at(labels, scores, 0.01),
        "tpr_at_5fpr": tpr_at(labels, scores, 0.05),
        "tpr_at_10fpr": tpr_at(labels, scores, 0.10),
    }


def stratified_bootstrap(labels, scores, n_boot, seed):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)
    names = ("auroc", "tpr_at_1fpr", "tpr_at_5fpr", "tpr_at_10fpr")
    values = {n: np.empty(n_boot, dtype=np.float64) for n in names}
    for b in tqdm(range(n_boot), desc="Bootstrap"):
        idx = np.r_[rng.choice(pos, len(pos), replace=True),
                    rng.choice(neg, len(neg), replace=True)]
        m = metrics(labels[idx], scores[idx])
        for name in names:
            values[name][b] = m[name]
    return {
        name: {
            "mean": float(v.mean()), "std": float(v.std(ddof=1)),
            "ci95_low": float(np.quantile(v, 0.025)),
            "ci95_high": float(np.quantile(v, 0.975)),
        } for name, v in values.items()
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device_index = int(args.device.split(":")[-1])
    torch.cuda.set_device(device_index)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    spec = MODEL_SPECS[args.model]
    # Fix the portable adapter's incorrect MiniCPM vision pattern as well.
    KINDS[args.model]["vision_pattern"] = spec["vision_pattern"]
    victim = GenVLMVictim(args.model, str(spec["model_dir"]), args.model, args.device)
    victim.load()
    selected, text_layers, vision_layers = select_last_layers(victim.model, spec, args.last_n)
    victim.mark_trainable(selected)
    split = load_split(args.seed, args.limit)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{args.mask_strategy}_tau{args.tau:g}_q{args.topq:g}"
    out = RESULT_ROOT / args.model / f"gradaudit_{run_tag}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "selected_parameters.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
    print(json.dumps({"model": args.model, "device": args.device,
                      "selected": len(selected), "text_layers": text_layers,
                      "vision_layers": vision_layers, "output": str(out)}, ensure_ascii=False), flush=True)

    failures = []
    counters = {}

    def gradient(rec, stage):
        try:
            with Image.open(rec.image) as im:
                image = im.convert("RGB")
            inputs = victim.build_lm_inputs(image, rec.prompt, rec.target,
                                            mask_mode="assistant_only")
            if inputs is None:
                raise RuntimeError("input construction returned None")
            # Training/backward does not use KV caching.  Explicitly disabling it
            # also avoids MiniCPM-V-2's legacy cache API incompatibility with
            # newer transformers releases; this does not change the logits/loss.
            inputs["use_cache"] = False
            victim.model.train()
            victim.model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = victim.model(**inputs)
            loss = output.loss if hasattr(output, "loss") else output[0]
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss: {loss.detach().item()}")
            loss.backward()
            wanted = set(selected)
            grads = {n: p.grad.detach().float() for n, p in victim.model.named_parameters()
                     if n in wanted and p.grad is not None}
            missing = len(selected) - len(grads)
            if not grads:
                raise RuntimeError("no selected parameter has a gradient")
            counters.setdefault(stage, {"attempted": 0, "valid": 0,
                                        "missing_gradient_params_total": 0,
                                        "missing_gradient_params_max": 0})
            counters[stage]["valid"] += 1
            counters[stage]["missing_gradient_params_total"] += missing
            counters[stage]["missing_gradient_params_max"] = max(
                counters[stage]["missing_gradient_params_max"], missing)
            return grads
        except Exception as exc:
            failures.append({"stage": stage, "sample_id": rec.uid, "label": rec.label,
                             "error_type": type(exc).__name__, "error": str(exc)[:1000]})
            if len(failures) <= 3:
                print(f"GRADIENT_FAILURE stage={stage} sample={rec.uid} "
                      f"type={type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return None
        finally:
            victim.model.zero_grad(set_to_none=True)
            victim.model.eval()

    def attempt(rec, stage):
        counters.setdefault(stage, {"attempted": 0, "valid": 0,
                                    "missing_gradient_params_total": 0,
                                    "missing_gradient_params_max": 0})
        counters[stage]["attempted"] += 1
        return gradient(rec, stage)

    ref = {}
    ref_valid = []
    for rec in tqdm(split.ref_members, desc="Reference member gradients"):
        gd = attempt(rec, "reference_member_mean")
        if gd is None:
            continue
        ref_valid.append(rec)
        for name, value in gd.items():
            ref[name] = ref.get(name, torch.zeros_like(value, device=value.device)) + value
    if not ref_valid:
        raise RuntimeError("All member reference gradients failed")
    ref = {name: value / len(ref_valid) for name, value in ref.items()}

    def group_sims(records, stage):
        row, col, valid = {}, {}, []
        for rec in tqdm(records, desc=stage):
            gd = attempt(rec, stage)
            if gd is None:
                continue
            valid.append(rec)
            for name, value in gd.items():
                if name not in ref:
                    continue
                rs, cs = row_col_cos(value, ref[name])
                row[name] = row.get(name, torch.zeros_like(rs)) + rs
                col[name] = col.get(name, torch.zeros_like(cs)) + cs
        if not valid:
            raise RuntimeError(f"All gradients failed in {stage}")
        return ({k: v / len(valid) for k, v in row.items()},
                {k: v / len(valid) for k, v in col.items()}, valid)

    mr, mc, valid_m = group_sims(split.ref_members, "reference_member_mask")
    nr, nc, valid_n = group_sims(split.ref_nonmembers, "reference_nonmember_mask")
    member_sims = {k: (mr[k], mc[k]) for k in mr if k in mc}
    nonmember_sims = {k: (nr[k], nc[k]) for k in nr if k in nc}
    row_mask, col_mask, sensitive = build_masks(
        ref, member_sims, nonmember_sims, tau=args.tau,
        strategy=args.mask_strategy, q=args.topq,
    )
    if not row_mask:
        raise RuntimeError("Sensitive mask is empty")

    score_rows = []
    def score_records(records, stage):
        for rec in tqdm(records, desc=stage):
            gd = attempt(rec, stage)
            if gd is None:
                continue
            pieces = []
            for name, value in gd.items():
                if name not in row_mask:
                    continue
                rs, cs = row_col_cos(value, ref[name])
                if row_mask[name].any():
                    pieces.append(rs[row_mask[name]])
                if col_mask[name].any():
                    pieces.append(cs[col_mask[name]])
            if not pieces:
                failures.append({"stage": stage, "sample_id": rec.uid, "label": rec.label,
                                 "error_type": "EmptyScore", "error": "no masked similarities"})
                continue
            # Scalar transfer only; all gradient/similarity work remains on CUDA.
            value = torch.cat(pieces).mean()
            if not torch.isfinite(value):
                failures.append({"stage": stage, "sample_id": rec.uid, "label": rec.label,
                                 "error_type": "NonFiniteScore", "error": str(value.item())})
                continue
            score_rows.append({"sample_id": rec.uid, "label": rec.label,
                               "score": float(value.item())})

    score_records(split.probe_members, "probe_member")
    score_records(split.probe_nonmembers, "probe_nonmember")
    labels = np.asarray([r["label"] for r in score_rows], dtype=np.int8)
    scores = np.asarray([r["score"] for r in score_rows], dtype=np.float64)
    if len(np.unique(labels)) != 2:
        raise RuntimeError("Valid probes do not contain both classes")
    point = metrics(labels, scores)
    boot = stratified_bootstrap(labels, scores, args.bootstrap, args.seed)

    with (out / "sample_scores.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "label", "score"])
        w.writeheader(); w.writerows(score_rows)
    with (out / "failed_samples.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["stage", "sample_id", "label", "error_type", "error"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(failures)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    with (out / "roc_curve.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["fpr", "tpr", "threshold"])
        w.writerows(zip(fpr, tpr, thresholds))
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.plot(fpr, tpr, linewidth=2.2, label=f"GradAudit (AUROC={point['auroc']:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="0.55", linewidth=1.2)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1))
    ax.grid(True, linestyle="--", alpha=.3); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "roc_curve.png", dpi=300); fig.savefig(out / "roc_curve.pdf")
    plt.close(fig)

    zero_counts = {
        "all": int(np.sum(scores == 0)),
        "member": int(np.sum(scores[labels == 1] == 0)),
        "nonmember": int(np.sum(scores[labels == 0] == 0)),
    }
    summary = {
        "model": args.model, "timestamp": stamp,
        "method": "GradAudit", "loss": "assistant_only_autoregressive",
        "strategy": "vision_text_last3_plus_projector_full_weights",
        "tau": args.tau, "mask_strategy": args.mask_strategy,
        "topq": args.topq, "aggregation": "mean",
        "seed": args.seed, "bootstrap_replicates": args.bootstrap,
        "text_layers": text_layers, "vision_layers": vision_layers,
        "selected_parameter_count": len(selected), "sensitive_dimensions": sensitive,
        "valid_probe_member": int(np.sum(labels == 1)),
        "valid_probe_nonmember": int(np.sum(labels == 0)),
        "failed_records": len(failures), "zero_score_counts": zero_counts,
        "stage_counts": counters, "point_estimate": point, "bootstrap": boot,
    }
    (out / "results_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    # A compact publication-facing source-data file without private file paths.
    with (out / "Source_Data.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "membership_label", "gradaudit_score"])
        w.writeheader()
        for r in score_rows:
            w.writerow({"sample_id": r["sample_id"], "membership_label": r["label"],
                        "gradaudit_score": r["score"]})
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
