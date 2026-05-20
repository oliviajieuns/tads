# 노드 가동 현황 — 2026-05-20

CIKM 2026 deadline (5/23) 대비 풀가동. 총 **10 노드** 동시 운용.

---

## 🎓 학습: thm_verify_7b (Figure 2 재현용)

LLaMA-2-7B + LoRA + Alpaca-GPT4 + 10% selection. Theorem 1 의 ε-stationarity 를 7B 스케일에서 입증하기 위함. 산출물: `metrics.jsonl` + `anchors/step_*.npy`.

| 노드 | Cell | Schedule | 목적 |
|---|---|---|---|
| **1107739-tads1** | cell B (TADS) | cosine, probe 2000 | Fig.2 빨간 곡선 (anchor 수렴) |
| **1107736-tads-a100-g20** | cell D (TADS) | constant lr, probe 2000 | Fig.2 보조 (per-step bound 검증) |
| **1108824-tads2-jieun** | no_anchor (λ=0) | cosine, matched control | Fig.2 회색 곡선 (anchor 안 쓰면 발산) |

**ETA**: ~14:00 today

**output**: `/group-volume/jieuns/tads-checkpoints/thm_verify_7b/llama2/<cell>/runs/<tag>/thm_verification/`

---

## 📊 Eval: Selection-Ratio Sweep

LLaMA-2-7B full FT 의 random_X% 셀에 대해 7 benchmarks 평가. paper 의 selection-ratio 효과 데이터.

**Benchmarks**: `mmlu, gsm8k, svamp, mbpp, tydiqa, xquad, bbh` (7개)

| 노드 | Cell | 체크포인트 |
|---|---|---|
| **1109346-tads2** | random_10 | `cikm2026_tads/checkpoints/7b_fullft/random_10/epoch_3/` |
| **1109345-tads1** | random_20 | `.../random_20/epoch_3/` |
| **1109349-tads3** | random_30 | `.../random_30/epoch_3/` |
| **1109350-tads4** | random_40 | `.../random_40/epoch_3/` |
| **1108825-tads3-jieun** | random_50 | `.../random_50/epoch_3/` |

**ETA**: ~13:30 today (셀당 ~8h, 병렬)

**output**: `/group-volume/jieuns/tads-eval-results/llama2/<cell>/runs/<tag>/` — 셀당 7 JSON + 1 summary = **8 JSON**

**스킵**: full_100 — 동료가 따로 처리, 결과 받아오기만

---

## 🔬 Scaling Study (Qwen2.5-14B)

14B 에서도 TADS 가 random 을 이긴다는 것을 보여주는 scaling 결과. paper App. F 14B section 강화.

LoRA (r=16, α=32, dropout=0.05, target=q/k/v/o/gate/up/down) + Alpaca-GPT4 + 10% selection + 3 epoch.

| 노드 | Cell | 상태 |
|---|---|---|
| **1109361-full100** | random_10 (14B) | 셋업 중 (omegaconf install + launch) |
| **(다음 노드)** | tads_10 (14B) | 80GB 노드 추가 잡으면 launch |

**ETA**: ~14h (내일 새벽~오전)

**output**: `/user-volume/jieuns/tads-checkpoints/scaling_14b/qwen25/<cell>/`

---

## 🆕 Base Baseline (W/O Finetuning)

Instruction tuning 전 base LLaMA-2-7B 의 raw 점수. paper Table 1 의 lower bound 행 ("Data-side references / W/O Finetuning").

| 노드 | Cell | 목적 |
|---|---|---|
| **1109402-tads5** | base eval (no FT) | instruction tuning 이전 lower bound |

**ETA**: ~8h

**output**: `/group-volume/jieuns/tads-eval-results/llama2/base_no_finetune/`

---

## 📊 총 가동 — 10 노드

```
학습   3 노드 (1107739, 1107736, 1108824)
Eval   5 노드 (1109346, 1109345, 1109349, 1109350, 1108825)
14B    1 노드 (1109361) + 곧 잡을 1 노드 = 2
Base   1 노드 (1109402)
────────────────────────────
총     10 노드
```

---

## 산출물 → Paper 매핑

| 결과 | 들어갈 위치 |
|---|---|
| thm_verify_7b 3 셀 | **Figure 2** (cumulative anchor drift @ 7B) |
| Random 5 셀 (10/20/30/40/50%) | **selection-ratio sweep table** |
| 14B random vs tads | **Table: scaling study** (App. F) |
| Base no-FT | **Main table 의 lower bound 행** |
| Full_100 (동료) | **Main table 의 upper bound 행** |

---

## 타임라인

| 시각 | 이정표 |
|---|---|
| **오늘 13:30** | Eval 5 셀 + base eval 일부 완료 |
| **오늘 18:00~24:00** | 학습 3 셀 + 추가 eval 완료 |
| **내일 새벽~오전** | 14B 2 셀 완료 |
| **5/21** | 결과 종합 + Figure 2 + 메인 테이블 채움 |
| **5/22** | paper 마무리, draft 정리 |
| **5/23** | 제출 ✅ |

---

## 모니터링 한 줄 (`~/check_all.sh` 추천)

```bash
echo "=== TRAIN ==="
for cell in B_probe2000_cosine D_probe2000_constant no_anchor_cosine; do
  LATEST=$(ls -td /group-volume/jieuns/tads-checkpoints/thm_verify_7b/llama2/$cell/runs/*/ 2>/dev/null | head -1)
  NANCH=$(ls $LATEST/thm_verification/anchors/step_*.npy 2>/dev/null | wc -l)
  NEPOCH=$(ls -d $LATEST/epoch_* 2>/dev/null | wc -l)
  printf "  %-25s anchors=%-4d epochs=%d/3\n" "$cell" "$NANCH" "$NEPOCH"
done

echo "=== EVAL ==="
for r in random_10 random_20 random_30 random_40 random_50 base_no_finetune; do
  LATEST=$(ls -td /group-volume/jieuns/tads-eval-results/llama2/$r/runs/*/ 2>/dev/null | head -1)
  N=$(ls $LATEST/*.json 2>/dev/null | wc -l)
  printf "  %-20s %d/8 json\n" "$r" "$N"
done

echo "=== 14B SCALING ==="
for cell in random_10 tads_10; do
  LATEST=$(ls -td /user-volume/jieuns/tads-checkpoints/scaling_14b/qwen25/$cell/runs/*/ 2>/dev/null | head -1)
  if [ -n "$LATEST" ]; then
    NEPOCH=$(ls -d $LATEST/epoch_* 2>/dev/null | wc -l)
    printf "  14b/%-15s epochs=%d/3\n" "$cell" "$NEPOCH"
  fi
done
```

---

## 환경 셋업 메모

**train/eval 공통 ENV (각 노드에서)**:
```bash
cd /group-volume/jieuns/tads_33m
source /group-volume/jieuns/llm-instruction-tuning/venv/bin/activate
source scripts/setup_env.sh
export HF_HOME=/group-volume/jieuns/tads-checkpoints/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=/group-volume/jieuns/tads-checkpoints/cache/datasets
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
```

**14B (user-volume 베이스, 1109361 처럼 tads_33m 안 보이는 노드용)**:
```bash
cd /user-volume/tads_space
source /user-volume/jieuns_venv/bin/activate
export MODEL_PATH_QWEN25_14B=/group-volume/jieuns/models/Qwen2.5-14B
export ALPACA_DATA_FILES='/group-volume/IT-datasets/alpaca_gpt4/data/*.json'
export OUTPUT_ROOT=/user-volume/jieuns/tads-checkpoints
export HF_HOME=$OUTPUT_ROOT/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
mkdir -p $HF_HOME/transformers $HF_HOME/datasets
```
