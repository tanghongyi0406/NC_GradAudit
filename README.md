# GradAudit

This repository provides the GradAudit implementation and frozen reproduction assets. The release includes model-specific environments, ready-to-use frozen datasets, the exact Qwen2-VL and InternVL3 LoRA adapters, optional fine-tuning code, and 10,000-replicate bootstrap evaluation.

## One-command reproduction

Install Conda, Git LFS, Git, and a CUDA 12.1-compatible NVIDIA driver. For gated Hugging Face assets, export `HF_TOKEN` after accepting the corresponding model terms. Then run:

```bash
git lfs pull
./quickstart.sh 0
```

The argument is the GPU identifier. The command creates independent environments, verifies and extracts the datasets, downloads pinned model assets, and runs the GradAudit evaluations sequentially. Outputs are written to `results/`.

## Recommended single-case reproduction

SmolVLM2 on COCO is the recommended reproduction case. It uses the frozen COCO evaluation archive and the pinned SmolVLM2 revision; no training or adapter is required.

```bash
./scripts/setup_envs.sh
./scripts/prepare_data.sh
conda run -n gradaudit-smolvlm2 python scripts/download_models.py --model smolvlm2
./scripts/run_one.sh smolvlm2 0
```

The corresponding paper result is **84.3 ± 1.0 AUROC**, **48.9 ± 2.5 TPR@5%FPR**, and **33.6 ± 3.3 TPR@1%FPR**, in percentage points. Small run-to-run numerical differences within the reported standard deviation are expected.

Successful execution writes the following files:

```text
results/smolvlm2/gradaudit_tau_or_topq_tau0.1_q0.9_YYYYMMDD_HHMMSS/
├── results_summary.json
├── sample_scores.csv
├── failed_samples.csv
├── roc_curve.csv
├── roc_curve.png
├── roc_curve.pdf
└── Source_Data.csv
```

`results_summary.json` contains the configuration, point estimates, failure and zero-score counts, and the 10,000-replicate bootstrap statistics. `sample_scores.csv` contains one score and membership label per valid audit sample, while `roc_curve.csv` and the image files contain the ROC data and plot. The summary has this abbreviated structure:

```json
{
  "model": "smolvlm2",
  "method": "GradAudit",
  "tau": 0.1,
  "mask_strategy": "tau_or_topq",
  "topq": 0.9,
  "bootstrap_replicates": 10000,
  "valid_probe_member": 800,
  "valid_probe_nonmember": 800,
  "point_estimate": {
    "auroc": 0.843,
    "tpr_at_5fpr": 0.489,
    "tpr_at_1fpr": 0.336
  },
  "bootstrap": {
    "auroc": {"mean": 0.843, "std": 0.010},
    "tpr_at_5fpr": {"mean": 0.489, "std": 0.025},
    "tpr_at_1fpr": {"mean": 0.336, "std": 0.033}
  }
}
```

This is an abbreviated example; the generated file contains the complete measured statistics.

## Standard evaluation and optional training

`quickstart.sh` is the complete standard reproduction entry point, `run_all.sh` runs all evaluations after setup, and `scripts/run_one.sh` runs one selected model. Valid model names are `smolvlm2`, `minicpm_v2`, `llava_med`, `minigpt_v2`, `qwen2vl_med`, and `internvl3_med`.

The standard evaluation uses the supplied Qwen2-VL and InternVL3 adapters. Re-training is **not required**. Programs under `training/` are optional and are not called by the standard reproduction commands.

## Data and adapters

`data_archives/` contains the directly usable frozen evaluation datasets. `scripts/prepare_data.sh` validates their SHA-256 digests and extracts them. `scripts/acquire_data.py` records the archive inventory and verifies local acquisition.

The exact fine-tuned weights are:

- `adapters/qwen_medtrinity_1k/adapter_model.safetensors`
- `adapters/internvl3_medtrinity_1k_1b/adapter_model.safetensors`

Their SHA-256 digests are recorded in `adapters/SHA256SUMS`. The standard reproduction path uses these adapters and does not fine-tune again. Optional training entry points are available in `training/`.

For the boundaries and provenance of the released method, assets, and third-party components, see [METHOD.md](METHOD.md), [PROVENANCE.md](PROVENANCE.md), [NOTICE](NOTICE), and [LICENSE](LICENSE).

## Storage and GPU requirements

Allow approximately 100 GB of free disk space for the pinned model downloads, extracted evaluation data, environments, caches, and result files. A CUDA-capable NVIDIA GPU is required for model execution. We recommend 48 GB of GPU memory for the complete one-command run. The smaller models can generally run with 16--24 GB, while LLaVA-Med and MiniGPT-v2 should be run on a 48 GB GPU to avoid CPU offloading.

## Repository layout

```text
src/                 GradAudit implementations
envs/                independent model environments
data_archives/       frozen datasets
adapters/            exact LoRA adapters
training/            optional adapter training programs
scripts/             setup, download, run, and evaluation utilities
```

## License

The original software and documentation are licensed under the MIT License. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Third-party models and datasets remain subject to their own terms.
