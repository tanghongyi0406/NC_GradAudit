#!/usr/bin/env python3
"""Document and validate construction of the frozen COCO MIA dataset.

COCO train and validation images were included in the training data used by
MiniGPT-v2, so neither split is an appropriate nonmember source.  We therefore
use COCO test images as nonmembers.  The official COCO test release contains
images but no public captions; captions for those images were generated with
GPT and frozen before auditing.  Member examples retain their COCO captions.
The released archive is the exact frozen dataset used by all COCO main-table
runs, including SmolVLM2 and MiniCPM-V-2.

This program does not call an online captioning service or silently regenerate
captions.  It validates the already frozen records so reproduction cannot
drift as an API or prompting setup changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    required = {0: 1000, 1: 1000}
    counts = {0: 0, 1: 0}
    for index, row in enumerate(rows):
        label = int(row["label"])
        if label not in counts:
            raise ValueError(f"record {index}: invalid label {label}")
        caption = row.get("caption")
        if isinstance(caption, list):
            caption = caption[0] if caption else ""
        if not str(caption).strip():
            raise ValueError(f"record {index}: missing frozen caption")
        if not str(row.get("image", "")).strip():
            raise ValueError(f"record {index}: missing image path")
        counts[label] += 1
    if counts != required:
        raise ValueError(f"expected {required}, found {counts}")
    print(f"Validated {len(rows)} frozen COCO records: {counts}")


if __name__ == "__main__":
    main()
