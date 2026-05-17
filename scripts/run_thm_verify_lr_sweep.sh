#!/bin/bash
# A1 cross-η verification — same D-cell setup (probe 2000, constant lr),
# but with four plateau-η values run concurrently on 4 GPUs.
# Test whether C_σ_est = mean(‖ΔΣ‖_F / η) is INDEPENDENT of η.
#
# If A1 holds:  C_σ_5e-6 ≈ C_σ_1e-5 ≈ C_σ_2e-5 ≈ C_σ_5e-5  (within ~20%)
# If A1 fails:  the four cells produce dispersed C_σ_est → A1 needs
#               paper-side weakening, which is a real theoretical finding.

set -u

REPO=/group-volume/jieuns/tads_v2
cd "$REPO"

source /group-volume/jieuns/llm-instruction-tuning/venv/bin/activate

COMMON_ENV='export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false'

LOG_DIR=$REPO/logs
mkdir -p "$LOG_DIR"

launch_cell () {
  local gpu=$1 lr_tag=$2
  local sess="thm_lr_${lr_tag}"
  local cfg="configs/experiments/thm_verify_lr_sweep/lr_${lr_tag}.yaml"
  local log="$LOG_DIR/thm_lr_${lr_tag}.log"

  tmux kill-session -t "$sess" 2>/dev/null
  tmux new -d -s "$sess"
  tmux send-keys -t "$sess" "source /group-volume/jieuns/llm-instruction-tuning/venv/bin/activate" C-m
  tmux send-keys -t "$sess" "cd $REPO" C-m
  tmux send-keys -t "$sess" "$COMMON_ENV" C-m
  tmux send-keys -t "$sess" \
    "CUDA_VISIBLE_DEVICES=${gpu} python -m tads.train --config ${cfg} --run_suffix=lr${lr_tag} 2>&1 | tee ${log}" C-m
  echo "GPU${gpu} → $sess → lr=${lr_tag} (log: $log)"
}

echo "============================================================"
echo "A1 cross-η verification launch | $(date)"
echo "============================================================"

launch_cell 0 5e-6
launch_cell 1 1e-5
launch_cell 2 2e-5
launch_cell 3 5e-5

cat <<'EOM'

All 4 cells launched. After ~1.5 h, analyse each and compare C_σ_est:

  for lr in 5e-6 1e-5 2e-5 5e-5; do
    RUN=$(ls -td /group-volume/jieuns/tads-checkpoints/light/thm_verify_lr_sweep/lr_${lr}/runs/*lr${lr}* 2>/dev/null | head -1)
    echo "=== lr=${lr} ==="
    python scripts/analyze_theorem.py --run_dir $RUN --skip_warmup_optsteps 50 --gamma_threshold 0.05
  done | tee logs/thm_lr_sweep_analysis.txt

A1 verdict for the paper:
  - All four cells return mode=constant and a C_σ within ±20% of the
    mean → A1 holds across magnitudes ⇒ strong A1 verification.
  - Dispersed C_σ → A1 needs probabilistic / filtered form.
EOM
