"""GradAudit implementation documentation."""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_metrics(labels, scores, allow_flip: bool = False) -> dict:
    """GradAudit implementation documentation."""
    labels = np.asarray(labels); scores = np.asarray(scores, dtype=float)
    auc = float(roc_auc_score(labels, scores))
    flipped = False
    if allow_flip and auc < 0.5:
        scores = -scores
        auc = 1.0 - auc
        flipped = True
    fpr, tpr, _ = roc_curve(labels, scores)
    out = {"auc": auc, "flipped": flipped}
    for target in (0.01, 0.05, 0.10):
        out[f"tpr@{int(target*100)}"] = float(tpr[int(np.abs(fpr - target).argmin())])
    return out


def bootstrap_auc(labels, scores, n_boot: int = 2000, seed: int = 42) -> dict:
    """GradAudit implementation documentation."""
    labels = np.asarray(labels); scores = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(labels); aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        lab = labels[idx]
        if lab.sum() == 0 or lab.sum() == n:   # Implementation note.
            continue
        aucs.append(roc_auc_score(lab, scores[idx]))
    aucs = np.array(aucs)
    return {
        "n_boot": int(len(aucs)),
        "mean":      float(aucs.mean()) if len(aucs) else None,
        "std":       float(aucs.std())  if len(aucs) else None,
        "ci95_low":  float(np.percentile(aucs, 2.5))  if len(aucs) else None,
        "ci95_high": float(np.percentile(aucs, 97.5)) if len(aucs) else None,
    }
