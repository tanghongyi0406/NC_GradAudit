"""GradAudit implementation documentation."""
import re
from typing import List

PROJECTOR_KEYS = ("mm_projector", "multi_modal_projector", "vision_proj",
                  "projector", "connector", "llama_proj")


def select_params(model, mode: str,
                  text_pattern: str = r"model\.layers\.(\d+)\.",
                  vision_pattern: str = r"visual\.blocks\.(\d+)\.",
                  last_n: int = 3) -> List[str]:
    """GradAudit implementation documentation."""
    named = list(model.named_parameters())
    if mode == "lora_all":
        return [n for n, p in named if "lora" in n.lower() and p.ndim >= 2]

    text_idx, vis_idx = set(), set()
    for n, _ in named:
        m = re.search(text_pattern, n)
        if m: text_idx.add(int(m.group(1)))
        m = re.search(vision_pattern, n)
        if m: vis_idx.add(int(m.group(1)))
    t_start = max(0, max(text_idx) - last_n + 1) if text_idx else 10**9
    v_start = max(0, max(vis_idx) - last_n + 1) if vis_idx else 10**9

    def in_last(name, pattern, start):
        m = re.search(pattern, name)
        return m is not None and int(m.group(1)) >= start

    sel = []
    for n, p in named:
        if p.ndim < 2:
            continue
        is_lora = "lora" in n.lower()
        is_proj = any(k in n for k in PROJECTOR_KEYS)
        if mode == "lora_last3_lora":
            if is_lora and (in_last(n, text_pattern, t_start)
                            or in_last(n, vision_pattern, v_start) or is_proj):
                sel.append(n)
        elif mode == "lora_last3_plus_base":
            if is_lora or in_last(n, text_pattern, t_start) or in_last(n, vision_pattern, v_start):
                sel.append(n)
        elif mode == "last3_full":
            if in_last(n, text_pattern, t_start) or in_last(n, vision_pattern, v_start) or is_proj:
                sel.append(n)
        elif mode == "all_full":
            if re.search(text_pattern, n) or re.search(vision_pattern, n) or is_proj:
                sel.append(n)
        else:
            raise ValueError(mode)
    return sel
