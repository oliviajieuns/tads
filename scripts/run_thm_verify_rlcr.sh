#!/bin/bash
# Run the RL+CR contrastive cell for fig:thm-main.
#
# Single 0.5B + LoRA run, one GPU, ~1.5 hours. Uses
# configs/experiments/thm_verify_05b_no_anchor.yaml -- anchor is
# computed by the verifier but disabled in the selection score
# (tads.use_anchor=false, tads.lam=0); lr is 2.5x the D cell to
# amplify volatility. The goal is a clearly-erratic d_step curve
# that exceeds the D-cell bound, not a strict re-verification.
#
# After completion, analyse the same way as the D cell:
#   python scripts/analyze_theorem.py \
#       --run_dir <output_root>/light/thm_verify_rlcr/no_anchor/runs/<tag>/ \
#       --skip_warmup_optsteps 250 --gamma_threshold 0.05 --robust
# and overlay this run's d_step against the D-cell bound by hand
# (or extend scripts/generate_paper_artifacts.py to read both).

set -u

REPO=/group-volume/jieuns/tads_v2
cd "$REPO"

source /group-volume/jieuns/llm-instruction-tuning/venv/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
       NUMEXPR_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false

GPU="${GPU:-0}"
SESS="${SESS:-thm_rlcr}"
LOG_DIR=$REPO/logs
mkdir -p "$LOG_DIR"

LOG=$LOG_DIR/thm_rlcr.log

tmux kill-session -t "$SESS" 2>/dev/null
tmux new -d -s "$SESS"
tmux send-keys -t "$SESS" "source /group-volume/jieuns/llm-instruction-tuning/venv/bin/activate" C-m
tmux send-keys -t "$SESS" "cd $REPO" C-m
tmux send-keys -t "$SESS" "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false" C-m
tmux send-keys -t "$SESS" \
  "CUDA_VISIBLE_DEVICES=$GPU python -m tads.train --config configs/experiments/thm_verify_05b_no_anchor.yaml --run_suffix=rlcr 2>&1 | tee $LOG" C-m

cat <<EOM

Launched RL+CR contrastive cell on GPU $GPU (tmux session: $SESS).
Wall time ~1.5 hours.

  tmux attach -t $SESS                    # watch live
  tail -f $LOG                            # or tail the log

After completion, output goes to
  /group-volume/jieuns/tads-checkpoints/light/thm_verify_rlcr/no_anchor/runs/<tag>/

EOM
