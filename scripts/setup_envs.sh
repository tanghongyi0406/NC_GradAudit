#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
command -v conda >/dev/null 2>&1 || { echo "Conda is required." >&2; exit 1; }
for spec in smolvlm2 minicpm_v2 llava minigpt qwen_med internvl3_med; do env_name="$(awk '/^name:/{print $2; exit}' "${ROOT}/envs/${spec}.yml")"; if conda env list | awk '{print $1}' | grep -qx "${env_name}"; then echo "Environment exists: ${env_name}"; else conda env create -f "${ROOT}/envs/${spec}.yml"; fi; done
