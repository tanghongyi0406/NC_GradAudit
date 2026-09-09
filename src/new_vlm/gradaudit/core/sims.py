"""GradAudit implementation documentation."""
from typing import Dict
import torch
import torch.nn.functional as F


def row_col_cos(g: torch.Tensor, ref: torch.Tensor):
    """GradAudit implementation documentation."""
    if g.device.type == "cpu" and torch.cuda.is_available():
        g = g.to("cuda", torch.float32, non_blocking=True)
        ref = ref.to("cuda", torch.float32, non_blocking=True)
    if g.ndim < 2:
        g = g.unsqueeze(0); ref = ref.unsqueeze(0)
    return (F.cosine_similarity(g, ref, dim=1),
            F.cosine_similarity(g.T, ref.T, dim=1))


def apply_mask(gap: torch.Tensor, tau: float, strategy: str,
               q: float = 0.80, min_keep_ratio: float = 0.20) -> torch.Tensor:
    if strategy == "fixed_tau":
        return gap > tau
    if strategy == "tau_or_topq":
        m = gap > tau
        if not m.any():
            m = gap >= torch.quantile(gap.float().flatten(), q).reshape(())
        return m
    if strategy == "tau_or_topq_union":
        tau_mask = gap > tau
        topq_mask = gap >= torch.quantile(gap.float().flatten(), q).reshape(())
        return tau_mask | topq_mask
    if strategy == "adaptive_keep":
        m = gap > tau
        if m.float().mean().item() < min_keep_ratio:
            m = gap >= torch.quantile(gap.float().flatten(), 1 - min_keep_ratio).reshape(())
        return m
    raise ValueError(strategy)


def build_masks(ref_grads: Dict[str, torch.Tensor],
                member_sims, nonmember_sims,
                tau: float = 0.10, strategy: str = "tau_or_topq",
                q: float = 0.80, min_keep_ratio: float = 0.20,
                require_positive: bool = False):
    """GradAudit implementation documentation."""
    row_masks, col_masks, total = {}, {}, 0
    for n, (mr, mc) in member_sims.items():
        if n not in nonmember_sims:
            continue
        nr, nc = nonmember_sims[n]
        r_gap = torch.nan_to_num(mr - nr, 0.0)
        c_gap = torch.nan_to_num(mc - nc, 0.0)
        rm = apply_mask(r_gap, tau, strategy, q, min_keep_ratio)
        cm = apply_mask(c_gap, tau, strategy, q, min_keep_ratio)
        if require_positive:
            rm &= r_gap > 0; cm &= c_gap > 0
        if rm.any() or cm.any():
            row_masks[n] = rm; col_masks[n] = cm
            total += int((rm.sum() + cm.sum()).item())
    return row_masks, col_masks, total
