#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; MODEL="${1:-}"; GPU="${2:-0}"
export CUDA_VISIBLE_DEVICES="${GPU}" TOKENIZERS_PARALLELISM=false
case "${MODEL}" in
  smolvlm2) conda run --no-capture-output -n gradaudit-smolvlm2 python "${ROOT}/src/smol_minicpm_gradaudit.py" --model smolvlm2 --device cuda:0 --tau 0.10 --mask-strategy tau_or_topq --topq 0.90 --bootstrap 10000 ;;
  minicpm_v2) conda run --no-capture-output -n gradaudit-minicpm-v2 python "${ROOT}/src/smol_minicpm_gradaudit.py" --model minicpmv2 --device cuda:0 --tau 0.10 --mask-strategy tau_or_topq --topq 0.90 --bootstrap 10000 ;;
  llava_med) conda run --no-capture-output -n gradaudit-llava python "${ROOT}/src/llava_med_gradaudit.py" --model_id "${ROOT}/models/llava-med-v1.5-mistral-7b-hf" --dataset_json "${ROOT}/data/pmcoa_roco_mia/dataset.json" --img_root "${ROOT}/data/pmcoa_roco_mia" --output_dir "${ROOT}/results/llava_med" ;;
  minigpt_v2) conda run --no-capture-output -n gradaudit-minigpt python "${ROOT}/src/minigpt_v2_gradaudit.py" --dataset_json "${ROOT}/data/minigpt_coco_gptcap_mia/dataset.json" --img_root "${ROOT}/data/minigpt_coco_gptcap_mia" --output_dir "${ROOT}/results/minigpt_v2" --repo_dir "${ROOT}/models/minigpt_v2/MiniGPT-4" --ckpt_dir "${ROOT}/models/minigpt_v2/checkpoints" --llm_dir "${ROOT}/models/minigpt_v2/llm" --hf_cache "${ROOT}/models/hf_cache" ;;
  qwen2vl_med) mkdir -p "${ROOT}/experiment_workspace/medtrinity_1k_2b/adapter_final"; cp -f "${ROOT}/adapters/qwen_medtrinity_1k/"* "${ROOT}/experiment_workspace/medtrinity_1k_2b/adapter_final/"; conda run --no-capture-output -n gradaudit-qwen-med python "${ROOT}/src/qwen2vl_medtrinity_gradaudit.py" --skip_train --base_model "${ROOT}/models/Qwen2-VL-2B-Instruct" --out_dir "${ROOT}/results/qwen2vl_med" --out_csv "${ROOT}/results/qwen2vl_med/scores.csv" ;;
  internvl3_med) conda run --no-capture-output -n gradaudit-internvl3-med python "${ROOT}/src/internvl3_medtrinity_gradaudit.py" --skip_train --base_model "${ROOT}/models/InternVL3-1B-hf" --adapter_store "${ROOT}/adapters" --results_dir "${ROOT}/results/internvl3_med" ;;
  *) echo "Usage: $0 {smolvlm2|minicpm_v2|llava_med|minigpt_v2|qwen2vl_med|internvl3_med} [gpu_id]" >&2; exit 2 ;;
esac
