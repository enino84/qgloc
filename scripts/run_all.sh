#!/usr/bin/env bash
# Run the suite.  bash scripts/run_all.sh [scale]
set -u
set -o pipefail   # otherwise a crashed experiment is reported as a success

SCALE="${1:-${SCALE:-smoke}}"
export SCALE PYTHONUNBUFFERED=1 MPLBACKEND=Agg
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${HERE}:${HERE}/experiments:${PYTHONPATH:-}"
RESULTS="${RESULTS_DIR:-${HERE}/results}"
mkdir -p "${RESULTS}/cache"
LOG="${RESULTS}/run_${SCALE}${SHARD_INDEX:+_shard$SHARD_INDEX}.log"

SELECTED="${EXPERIMENTS:-exp00_setup exp01_assignment}"
echo "=== qgloc  scale=${SCALE}  $(date -Is) ===" | tee -a "${LOG}"
echo "experiments: ${SELECTED}"                   | tee -a "${LOG}"

FAILED=""
for exp in ${SELECTED}; do
  echo "" | tee -a "${LOG}"; echo ">>> ${exp}" | tee -a "${LOG}"
  if python3 "${HERE}/experiments/${exp}.py" "${SCALE}" 2>&1 | tee -a "${LOG}"; then
    echo "<<< ${exp} ok" | tee -a "${LOG}"
  else
    echo "<<< ${exp} FAILED" | tee -a "${LOG}"; FAILED="${FAILED} ${exp}"
  fi
done
[ -n "${FAILED}" ] && { echo "FAILED:${FAILED}" | tee -a "${LOG}"; exit 1; }
echo "all done $(date -Is)" | tee -a "${LOG}"
