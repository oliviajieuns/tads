#!/bin/bash
# evol_7b / qwen25 — Chain B (GPU 4-7) — full_100 → data_agent_10
#
# Sequential 4-GPU DDP chain on the second half of the node. Disjoint with
# chainA.sh (GPUs 0-3, port 29501) — the two can run simultaneously.
#
# Ordering rationale: heaviest cell first (full_100 trains on the full 70K
# samples × 3 epochs ≈ 3-4h), then data_agent_10 (= TADS λ=0 ablation
# companion). This minimises the chance that a node reclaim mid-chain costs
# us the longest job.
#
# Failure policy: identical to chainA — abort on first non-zero exit.

set -u

REPO=/group-volume/jieuns/tads_v2
cd "$REPO"

source /group-volume/jieuns/llm-instruction-tuning/venv/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
       NUMEXPR_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false

# Defaults assume chain B runs alongside chain A on the same 8-GPU node:
# GPUs 4-7, port 29502. Override per-node via env, e.g. on a 4-GPU node:
#   GPUS=0,1,2,3 PORT=29502 bash scripts/run_evol_qwen_chainB.sh
GPUS="${GPUS:-4,5,6,7}"
PORT="${PORT:-29502}"
NPROC="$(echo "$GPUS" | awk -F',' '{print NF}')"
METHODS=(full_100 data_agent_10)
LOG_DIR=$REPO/logs
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Chain B start | GPUs=$GPUS | port=$PORT | methods=${METHODS[*]}"
echo "Repo=$REPO | $(date)"
echo "============================================================"

for method in "${METHODS[@]}"; do
  TS=$(date +%Y%m%d_%H%M%S)
  log=$LOG_DIR/evol_qwen_chainB_${method}_${TS}.log
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
    echo "!!! $method failed (exit=$rc) — aborting chain B" | tee -a "$log"
    exit "$rc"
  fi
  sleep 30
done

echo
echo "============================================================"
echo "Chain B complete | $(date)"
echo "============================================================"
