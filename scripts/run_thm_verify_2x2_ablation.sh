#!/bin/bash
# 2×2 ablation for App. F Theorem 1 verification.
#
# Pins one config per GPU (single-GPU jobs, no DDP). Each cell runs
# concurrently so the whole 2×2 grid finishes in the wall time of the
# slowest cell (~1.5 h on a 0.5B model). Tmux sessions are named so
# attach/detach is straightforward:
#
#   tmux attach -t thm_abl_A
#   tmux attach -t thm_abl_B    ...
#
# All four cells write under <output_root>/light/thm_verify_ablation/
# <cell>/runs/<timestamp>/thm_verification/ — analyse each independently
# with scripts/analyze_theorem.py and compare PASS_FAIL.json.

set -u

REPO=/group-volume/jieuns/tads_v2
cd "$REPO"

source /group-volume/jieuns/llm-instruction-tuning/venv/bin/activate

# Thread caps — same libgomp safety as the evol chains.
COMMON_ENV='export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false'

LOG_DIR=$REPO/logs
mkdir -p "$LOG_DIR"

launch_cell () {
  local gpu=$1 letter=$2 cfg_stem=$3
  local sess="thm_abl_${letter}"
  local cfg="configs/experiments/thm_verify_ablation/${cfg_stem}.yaml"
  local log="$LOG_DIR/thm_abl_${letter}_${cfg_stem}.log"

  tmux kill-session -t "$sess" 2>/dev/null
  tmux new -d -s "$sess"
  tmux send-keys -t "$sess" "source /group-volume/jieuns/llm-instruction-tuning/venv/bin/activate" C-m
  tmux send-keys -t "$sess" "cd $REPO" C-m
  tmux send-keys -t "$sess" "$COMMON_ENV" C-m
  tmux send-keys -t "$sess" \
    "CUDA_VISIBLE_DEVICES=${gpu} python -m tads.train --config ${cfg} --run_suffix=abl${letter} 2>&1 | tee ${log}" C-m
  echo "GPU${gpu} → $sess → $cfg (log: $log)"
}

echo "============================================================"
echo "2x2 ablation launch | $(date)"
echo "============================================================"

launch_cell 0 A A_probe256_cosine
launch_cell 1 B B_probe2000_cosine
launch_cell 2 C C_probe256_constant
launch_cell 3 D D_probe2000_constant

echo
echo "All 4 cells launched. Monitor with:"
echo "  for s in thm_abl_A thm_abl_B thm_abl_C thm_abl_D; do"
echo "    echo \"=== \$s ===\"; tail -n 5 logs/\${s}_*.log | tail -n 5"
echo "  done"
echo
echo "After they finish, analyse each:"
echo "  for c in A_probe256_cosine B_probe2000_cosine C_probe256_constant D_probe2000_constant; do"
echo "    RUN=\$(ls -td /group-volume/jieuns/tads-checkpoints/light/thm_verify_ablation/\${c}/runs/*ablA* 2>/dev/null | head -1)"
echo "    python scripts/analyze_theorem.py --run_dir \$RUN --skip_warmup_optsteps 50 --gamma_threshold 0.05"
echo "  done"
