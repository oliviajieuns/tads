#!/usr/bin/env bash
# Multi-node sanity check for the 2026-05-21 run-fleet.
# Run from any node that has /group-volume + /user-volume mounted.
# Reports: alive-or-dead, latest progress line, output file count.

set -u

now=$(date "+%Y-%m-%d %H:%M:%S")
echo "===== check_all $now ====="

# ---- helper: latest log line under a run dir ----
latest_log_line() {
  local d="$1"
  if [ -d "$d" ]; then
    local f
    f=$(find "$d" -maxdepth 3 -name "*.log" -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | head -1 | cut -d' ' -f2-)
    if [ -n "$f" ]; then
      tail -1 "$f" 2>/dev/null | cut -c1-200
    else
      echo "(no .log under $d)"
    fi
  else
    echo "(missing $d)"
  fi
}

# ---- helper: count JSON files for an eval cell ----
count_json() {
  local d="$1"
  find "$d" -maxdepth 4 -name "*.json" 2>/dev/null | wc -l
}

# =========================================================================
# 1. Evol full_100 (1109402-tads5) — eval +3 bench
# =========================================================================
echo ""
echo "--- 1109402 | Evol full_100 (eval +svamp/mbpp/xquad) ---"
D=/group-volume/jieuns/tads-eval-results/llama2_evol/full_100
LAT=$(ls -td $D/runs/*/ 2>/dev/null | head -1)
echo "  latest run: ${LAT:-NONE}"
echo "  JSON cnt  : $(count_json "$LAT")  (target: 3 + summary = 4)"
echo "  last line : $(latest_log_line "$LAT")"

# =========================================================================
# 2. Evol random_10 (1109879) — eval 8 bench
# =========================================================================
echo ""
echo "--- 1109879 | Evol random_10 (eval 8 bench) ---"
D=/group-volume/jieuns/tads-eval-results/llama2_evol/random_10
LAT=$(ls -td $D/runs/*/ 2>/dev/null | head -1)
echo "  latest run: ${LAT:-NONE}"
echo "  JSON cnt  : $(count_json "$LAT")  (target: 8 + summary = 9)"
echo "  last line : $(latest_log_line "$LAT")"

# =========================================================================
# 3. Evol tads_10 (1109881) — training (full FT)
# =========================================================================
echo ""
echo "--- 1109881 | Evol tads_10 (training, full FT) ---"
D=/group-volume/jieuns/tads-checkpoints/evol_7b/llama2/tads_10
LAT=$(ls -td $D/runs/*/ 2>/dev/null | head -1)
NEPOCH=0; [ -n "$LAT" ] && NEPOCH=$(find "$LAT" -maxdepth 1 -type d -name "epoch_*" 2>/dev/null | wc -l)
NSEL=0;   [ -n "$LAT" ] && NSEL=$(find "$LAT" -maxdepth 1 -name "selected_indices_epoch*.json" 2>/dev/null | wc -l)
echo "  latest run    : ${LAT:-NONE}"
echo "  epoch dirs    : $NEPOCH / 3"
echo "  selected idx  : $NSEL"
# launch log lives at $D/launch_*.log
LL=$(ls -t $D/launch_*.log 2>/dev/null | head -1)
echo "  launch log    : ${LL:-NONE}"
if [ -n "$LL" ]; then
  echo "  last 2 lines  :"
  tail -2 "$LL" 2>/dev/null | sed 's/^/    /'
fi

# =========================================================================
# 4. 14B base no-FT (1109350-tads4) — eval 8 bench
# =========================================================================
echo ""
echo "--- 1109350 | 14B base no-FT (eval 8 bench) ---"
D=/group-volume/jieuns/tads-eval-results/qwen25-14b/base_no_finetune
LAT=$(ls -td $D/runs/*/runs/*/ 2>/dev/null | head -1)   # eval.py uses runs/<tag>/runs/<eval_tag>/
[ -z "$LAT" ] && LAT=$(ls -td $D/runs/*/ 2>/dev/null | head -1)
echo "  latest run: ${LAT:-NONE}"
echo "  JSON cnt  : $(count_json "$LAT")  (target: 8 + summary = 9)"
echo "  last line : $(latest_log_line "$LAT")"

# =========================================================================
# 5. 14B tads_10 (1109816-test14b) — training
# =========================================================================
echo ""
echo "--- 1109816 | 14B tads_10 (training, LoRA) ---"
D=/group-volume/jieuns/tads-checkpoints/scaling_14b/qwen25/tads_10
LAT=$(ls -td $D/runs/*/ 2>/dev/null | head -1)
NEPOCH=0; [ -n "$LAT" ] && NEPOCH=$(find "$LAT" -maxdepth 1 -type d -name "epoch_*" 2>/dev/null | wc -l)
echo "  latest run: ${LAT:-NONE}"
echo "  epoch dirs: $NEPOCH / 3"
echo "  last line : $(latest_log_line "$LAT")"

# =========================================================================
# 6. 14B random_10 (1109349-tads3) — training (재시도)
# =========================================================================
echo ""
echo "--- 1109349 | 14B random_10 (re-training) ---"
D=/group-volume/jieuns/tads-checkpoints/scaling_14b/qwen25/random_10
LAT=$(ls -td $D/runs/*/ 2>/dev/null | head -1)
NEPOCH=0; [ -n "$LAT" ] && NEPOCH=$(find "$LAT" -maxdepth 1 -type d -name "epoch_*" 2>/dev/null | wc -l)
EL=""; [ -n "$LAT" ] && [ -d "$LAT/epoch_last" ] && EL="✅"
echo "  latest run: ${LAT:-NONE}"
echo "  epoch dirs: $NEPOCH / 3   epoch_last: $EL"
echo "  last line : $(latest_log_line "$LAT")"

# =========================================================================
# 7. 14B random_10 eval (whichever node) — uses old user-volume ckpt
# =========================================================================
echo ""
echo "--- 1109879 (or other) | 14B random_10 (eval) ---"
D=/group-volume/jieuns/tads-eval-results/qwen25-14b/random_10
LAT=$(ls -td $D/runs/*/ 2>/dev/null | head -1)
echo "  latest run: ${LAT:-NONE}"
echo "  JSON cnt  : $(count_json "$LAT")  (target: 8 + summary = 9)"
echo "  last line : $(latest_log_line "$LAT")"

# =========================================================================
# 8. 7B residual (random_50 humaneval, random_20 BBH) — leftover patches
# =========================================================================
echo ""
echo "--- 7B 잔여 패치 (random_50 humaneval / random_20 bbh) ---"
for cell in random_50 random_20; do
  D=/group-volume/jieuns/tads-eval-results/llama2/$cell
  LAT=$(ls -td $D/runs/*/ 2>/dev/null | head -1)
  N=$(count_json "$LAT")
  printf "  %-12s  JSON cnt=%d   run=%s\n" "$cell" "$N" "${LAT:-NONE}"
done

# =========================================================================
# 9. 7B full_100 TydiQA (동료 노드)
# =========================================================================
echo ""
echo "--- 7B full_100 TydiQA F1 (동료 노드, 진행 중) ---"
D=/group-volume/jieuns/tads-eval-results/llama2/full_100
LAT=$(ls -td $D/runs/*/ 2>/dev/null | head -1)
echo "  latest run: ${LAT:-NONE}"
echo "  last line : $(latest_log_line "$LAT")"

echo ""
echo "===== done ====="
