"""GradAudit implementation documentation."""
import json, os, random, re
from dataclasses import dataclass, field
from typing import List, Optional

# Implementation note.
# Implementation note.
PROMPT_DEFAULT = ("You are a medical vision-language model. Please describe the "
                  "medical image with detailed, factual findings.")

_IMAGE_TOKEN_RE = re.compile(r"<\s*image\s*>", flags=re.IGNORECASE)

def strip_image_token(p: str) -> str:
    if not isinstance(p, str):
        return ""
    p = _IMAGE_TOKEN_RE.sub("", p).replace("\n", " ").strip()
    return re.sub(r"\s+", " ", p).strip()


@dataclass
class MIARecord:
    image: str          # Implementation note.
    target: str         # Implementation note.
    label: int          # 1=member 0=nonmember
    prompt: str = PROMPT_DEFAULT  # Implementation note.
    uid: str = ""       # Implementation note.
    pair_id: str = ""   # Implementation note.


def load_records(json_path: str, img_root: Optional[str], label: int,
                 default_prompt: str = PROMPT_DEFAULT) -> List[MIARecord]:
    """GradAudit implementation documentation."""
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "data" in raw:
        raw = raw["data"]
    out = []
    for r in raw:
        if "messages" in r and "images" in r:
            rel = r["images"][0]
            msgs = r["messages"]
            prompt = strip_image_token(msgs[0]["content"]) or default_prompt
            target = msgs[-1]["content"]
            if isinstance(target, list):
                target = " ".join(seg.get("text", "") if isinstance(seg, dict) else str(seg)
                                  for seg in target)
        else:
            rel = r.get("image") or r.get("local_image_path") or r.get("image_path")
            cap = r.get("caption", "")
            if isinstance(cap, list):
                cap = cap[0] if cap else ""
            prompt, target = default_prompt, str(cap)
        path = rel if (img_root is None or os.path.isabs(rel)) else os.path.join(img_root, rel)
        out.append(MIARecord(image=path, target=str(target), label=label,
                             prompt=prompt, uid=rel))
    return out


@dataclass
class Split:
    train: List[MIARecord]
    ref_members: List[MIARecord]
    ref_nonmembers: List[MIARecord]
    probe_members: List[MIARecord]
    probe_nonmembers: List[MIARecord]
    fingerprint: dict = field(default_factory=dict)


def make_split(members: List[MIARecord], nonmembers: List[MIARecord],
               seed: int = 42, train_size: int = 1000,
               ref_size: int = 200, probe_size: int = 800) -> Split:
    """GradAudit implementation documentation."""
    cs = seed + 123
    members = list(members); nonmembers = list(nonmembers)
    random.Random(cs).shuffle(members)
    random.Random(cs).shuffle(nonmembers)
    train = members[:train_size]
    mia_m = random.Random(cs).sample(train, min(train_size, len(train)))
    mia_n = nonmembers[:train_size]
    fp = {"config_seed": cs, "train_size": len(train),
          "ref_member_0": mia_m[0].uid if mia_m else "",
          "probe_member_0_idx_ref": mia_m[ref_size].uid if len(mia_m) > ref_size else "",
          "ref_nonmember_0": mia_n[0].uid if mia_n else ""}
    return Split(train=train,
                 ref_members=mia_m[:ref_size],
                 ref_nonmembers=mia_n[:ref_size],
                 probe_members=mia_m[ref_size:ref_size + probe_size],
                 probe_nonmembers=mia_n[ref_size:ref_size + probe_size],
                 fingerprint=fp)


def make_split_labeled(records: List[MIARecord], seed: int = 42, n_each: int = 1000,
                       ref_size: int = 200, probe_size: int = 800) -> Split:
    """GradAudit implementation documentation."""
    ms = [r for r in records if r.label == 1]
    ns = [r for r in records if r.label == 0]
    random.Random(seed).shuffle(ms)
    random.Random(seed).shuffle(ns)
    return make_split(ms[:n_each], ns[:n_each], seed=seed,
                      train_size=n_each, ref_size=ref_size, probe_size=probe_size)
