#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; GPU="${1:-0}"
"${ROOT}/scripts/setup_envs.sh"
"${ROOT}/scripts/prepare_data.sh"
conda run --no-capture-output -n gradaudit-qwen-med python "${ROOT}/scripts/download_models.py" --model all
"${ROOT}/run_all.sh" "${GPU}"
