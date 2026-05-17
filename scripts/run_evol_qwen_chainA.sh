#!/bin/bash
# evol_7b / qwen25 — Chain A (GPU 0-3) — random_10 → tads_10
#
# Sequential 4-GPU DDP chain. Companion to chainB.sh; the two run in parallel
# on disjoint GPU sets (0-3 vs 4-7) and disjoint master ports.
#
# Ordering rationale: random_10 first (fast, lower-bound baseline), then
# tads_10 (the headline method). This way the most important comparison
# point (TADS vs Random) lands at chain-A completion (~4.5h).
#
# Failure policy: if any cell exits non-zero, abort the rest of the chain
# rather than cascade-corrupting downstream cells. The caller can fix and
# re-launch — the timestamp tag means a clean retry starts a fresh
# runs/<new_ts>/ without touching prior work.

set -u

REPO=/group-volume/jieuns/tads_v2
cd "$REPO"

# venv + thread caps (libgomp safety; see docs/experiments/evol_qwen25_7b.md §7)
source /group-volume/jieuns/llm-instruction-tuning/venv/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
       NUMEXPR_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false

# Defaults: GPU 0-3, port 29501. Override per-node via env, e.g.
#   GPUS=0,1,2,3 PORT=29501 bash scripts/run_evol_qwen_chainA.sh
GPUS="${GPUS:-0,1,2,3}"
PORT="${PORT:-29501}"
NPROC="$(echo "$GPUS" | awk -F',' '{print NF}')"
METHODS=(random_10 tads_10)
LOG_DIR=$REPO/logs
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Chain A start | GPUs=$GPUS | port=$PORT | methods=${METHODS[*]}"
echo "Repo=$REPO | $(date)"
echo "============================================================"

for method in "${METHODS[@]}"; do
  TS=$(date +%Y%m%d_%H%M%S)
  log=$LOG_DIR/evol_qwen_chainA_${method}_${TS}.log
  echo
  echo "=== START $method at $(date) ===" | tee -a "$log"
  CUDA_VISIBLE_DEVICES=$GPUS torchrun \
      --nproc_per_node=$NPROC \
      --master_port=$PORT \
      -m tads.train \
      --config configs/experiments/evol_7b/qwen25/${method}.yaml \
      2>&1 | tee -a "$log"
  rc=${PIPESTATUS[0]}
  echo "=== DONE  $method (exit=$rc) at $(date) ===" | tee -a "$log"
  if [ "$rc" -ne 0 ]; then
    echo "!!! $method failed (exit=$rc) — aborting chain A" | tee -a "$log"
    exit "$rc"
  fi
  sleep 30   # give NCCL / cuda allocator time to release before the next cell
done

echo
echo "============================================================"
echo "Chain A complete | $(date)"
echo "============================================================"
