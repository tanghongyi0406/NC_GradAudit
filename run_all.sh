#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; GPU="${1:-0}"
"${ROOT}/scripts/prepare_data.sh"
for model in smolvlm2 minicpm_v2 llava_med minigpt_v2 qwen2vl_med internvl3_med; do echo "Running ${model} on GPU ${GPU}"; "${ROOT}/scripts/run_one.sh" "${model}" "${GPU}"; done
echo "All six GradAudit runs completed under ${ROOT}/results."
