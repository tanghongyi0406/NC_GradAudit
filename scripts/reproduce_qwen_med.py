#!/usr/bin/env python3
"""Run the Qwen2-VL/MedTrinity GradAudit experiment.

This uses the released checkpoint-315 adapter and evaluates the fixed
reference/probe partition defined by the experiment configuration.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMPL = ROOT / "src/qwen2vl_medtrinity_gradaudit.py"
ADAPTER = ROOT / "adapters/qwen_medtrinity_1k"


def load_impl():
    spec = importlib.util.spec_from_file_location("qwen_med_impl", IMPL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/qwen2vl_med")
    args = parser.parse_args()

    impl = load_impl()
    reference_size = 200
    impl.CALIB_MEM = reference_size
    impl.CALIB_NON = reference_size

    original_loader = impl.load_json_list

    def fixed_probe_loader(path):
        records = original_loader(path)
        impl.random.Random(impl.ATTACK_SEED).shuffle(records)
        reference = records[:reference_size]
        probe = records[200:1000]
        used_ids = {id(item) for item in reference + probe}
        padding = [item for item in records if id(item) not in used_ids]
        desired = (reference + probe + padding)[:1000]
        permutation = list(range(len(desired)))
        impl.random.Random(impl.ATTACK_SEED).shuffle(permutation)
        restored = [None] * len(desired)
        for destination, source in enumerate(permutation):
            restored[source] = desired[destination]
        return restored

    impl.load_json_list = fixed_probe_loader
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_info = {
        "base_model": str(ROOT / "models/Qwen2-VL-2B-Instruct"),
        "lora_dir": str(ADAPTER),
    }
    config = dict(impl.CONFIG)
    config["id"] = "medtrinity_1k_main_table_ref200"
    impl.run_gradsafe_attack(
        model_info,
        str(args.output_dir / "scores.csv"),
        str(args.output_dir),
        config,
    )


if __name__ == "__main__":
    main()
