#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVES="${ROOT}/data_archives"
DATA="${ROOT}/data"

cd "${ARCHIVES}"
sha256sum --check SHA256SUMS
mkdir -p "${DATA}"
for archive in *.tar.gz; do
  dataset="${archive%.tar.gz}"
  target="${DATA}"
  [[ "${dataset}" == "medtrinity_mia" ]] && target="${DATA}/frozen_dataset"
  mkdir -p "${target}"
  if [[ -f "${target}/${dataset}/dataset.json" ]] || \
     [[ -f "${target}/${dataset}/member_eval_1000.json" ]]; then
    echo "Dataset already prepared: ${dataset}"
  else
    echo "Extracting ${archive}"
    tar -xzf "${archive}" -C "${target}"
  fi
done
python "${ROOT}/scripts/build_coco_mia_dataset.py" \
  "${DATA}/minigpt_coco_gptcap_mia/dataset.json"
echo "Frozen datasets are ready under ${DATA}."
