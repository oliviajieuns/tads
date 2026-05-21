# 노드 가동 현황 — 2026-05-21

CIKM 2026 deadline (5/23) D-2. 14B scaling 마무리 + 14B base baseline + **Evol-Instruct appendix 풀가동**.

---

## 🆕 14B Base Baseline (Qwen2.5-14B, no FT) — **NEW**

Instruction tuning 전 Qwen2.5-14B 의 raw 점수. 14B scaling 표의 lower bound 행.

| 노드 | Cell | 모델 / mode | 목적 |
|---|---|---|---|
| **1109350-tads4** | base eval (no FT) | Qwen2.5-14B / full 로딩 (LoRA 없음) | 14B scaling 표 lower bound |

**ETA**: ~6-9h (8 bench, bf16, A100-80GB) → 오늘 14:00~17:00 완료 예상

**output**:
```
/group-volume/jieuns/tads-eval-results/qwen25-14b/base_no_finetune/runs/20260521_080510/runs/<eval_tag>/
  qwen25_random_10-{mmlu,bbh,gsm8k,svamp,humaneval,mbpp,tydiqa,xquad}.json
  qwen25_random_10-eval_summary.json
```

**launch**: `tads.eval --ckpt /group-volume/jieuns/models/Qwen2.5-14B --training_mode full --benchmarks mmlu,bbh,gsm8k,svamp,humaneval,mbpp,tydiqa,xquad`

---

## 🔬 Scaling Study (Qwen2.5-14B + LoRA)

14B 에서도 TADS 가 random 을 이긴다는 결과. paper App. F.

| 노드 | Cell | 상태 |
|---|---|---|
| **1109816-test14b** | tads_10 학습 | epoch 1 collect 62.5% (오늘 03:20 시작) — ETA 5/22 새벽 |
| **1109349-tads3** | random_10 학습 (재시도) | epoch_last ✅ — (옛 epoch_3 있어서 사실 불필요) |
| **1109879-qwen14b-eval** | random_10 eval (옛 epoch_3 사용) | 2/8 bench done (MMLU 79.82, GSM8K 83.85), SVAMP 진행 중 |

옛 random_10 ckpt: `/user-volume/jieuns/tads-checkpoints/scaling_14b/qwen25/random_10/epoch_3` (LoRA 800M, 5/17 완료)

**output**:
- 학습: `/group-volume/jieuns/tads-checkpoints/scaling_14b/qwen25/<cell>/runs/<tag>/`
- eval: `/group-volume/jieuns/tads-eval-results/qwen25-14b/<cell>/runs/<tag>/`

⚠️ 14B full_100 체크포인트 **없음** — 표는 base / random_10 / tads_10 3행 구성.

---

## 📊 7B Eval: Selection-Ratio Sweep — 진행 중 / 마무리 단계

| 노드 | Cell | 상태 |
|---|---|---|
| **1109346-tads2** | random_10 | 완료 |
| **1109345-tads1** | random_20 → BBH 재실행 | BBH 단일 진행 중 |
| **1109349-tads3** | random_30 | 완료 → 14B random_10 재학습으로 전환 |
| **1109350-tads4** | random_40 | 완료 → **14B base eval 로 전환** ✅ |
| **1108825-tads3-jieun** | random_50 | HumanEval 재실행 진행 중 |

**output**: `/group-volume/jieuns/tads-eval-results/llama2/<cell>/runs/<tag>/`

---

## 🆕 7B Base Baseline (no FT) — 완료

| 노드 | Cell | 상태 |
|---|---|---|
| ~~1109402-tads5~~ | base eval (LLaMA-2-7B, no FT) | ✅ 8 bench 완료 (MBPP=0 footnote 처리됨) → **Evol full_100 +3 bench 로 전환** |

---

## 🆕 Evol-Instruct Appendix (LLaMA-2-7B × WizardLM Evol-Instruct 70K) — **NEW**

dataset: `/group-volume/IT-datasets/wizardlm_evol_instruct_70k/alpaca_evol_instruct_70k.json` (70K)  
model: `/group-volume/nait-models/Llama-2-7b-hf`, mode: **full FT**, prompt_style: `llama_user_assistant`

| 노드 | Cell | 작업 | 상태 / ETA |
|---|---|---|---|
| **1109402-tads5** | full_100 (eval) | svamp/mbpp/xquad 3 bench 추가 | ETA ~1-2h → 5/21 오전 완료 |
| **1109879** | random_10 (eval) | 8 bench 처음부터 | ETA ~4-6h → 5/21 오후 완료 |
| **1109881** | tads_10 (학습) | full FT, bs=4 ga=32 (effective 128), 3 epoch | ETA ~24-30h → **5/22 오후~저녁** |

학습 log: `/group-volume/jieuns/tads-checkpoints/evol_7b/llama2/tads_10/launch_20260521_081801.log`

**기존 (5/14 완료)**: full_100 (5/8 bench 옛 결과, 동료 본), random_10 (학습 ✅ eval 0), data_agent_10 (학습 ❌)

⚠️ **time risk**: 1-GPU full FT 7B 가 원본 4-GPU DDP 보다 4x 느림. 5/22 새벽까지 epoch 1 도 못 끝나면 LoRA 모드로 switch 결정.

**output**:
- eval: `/group-volume/jieuns/tads-eval-results/llama2_evol/<cell>/runs/<tag>/`
- 학습: `/group-volume/jieuns/tads-checkpoints/evol_7b/llama2/tads_10/runs/<tag>/`

---

## 🎓 thm_verify_7b — 부분 데이터로 drop 확정

3 셀 (B/D/no_anchor) 모두 step 975 (mid-epoch-2) 에서 멈춤 — verifier `track_delta_sigma=True` 메모리 누적이 원인. 7B Fig 2 재현 미실시. Figure 2 는 0.5B 본 유지.

---

## 📊 산출물 → Paper 매핑 (5/21 기준)

| 결과 | 들어갈 위치 |
|---|---|
| 7B base no-FT | Main table lower bound 행 ✅ |
| 7B random 10/20/30/40/50% | Main R10 + Appendix R20-R50 ratio sweep ✅ |
| 7B full_100 (동료) | Main table upper bound — TydiQA F1 진행 중 (88%) |
| **14B base no-FT** | **scaling 표 lower bound (신규)** |
| 14B random_10 (옛 ckpt) | scaling 표 random 행 |
| 14B tads_10 (학습 후) | scaling 표 TADS 행 |
| **Evol-Instruct full_100** | **Appendix Evol upper bound** (8 bench 완성 후) |
| **Evol-Instruct random_10** | **Appendix Evol random 행** |
| **Evol-Instruct tads_10** | **Appendix Evol TADS 행** (5/22 학습 완료 후) |

---

## 타임라인

| 시각 | 이정표 |
|---|---|
| **오늘 09:30~10:30** | Evol full_100 +3 bench (1109402) 완료 |
| **오늘 14:00~17:00** | 14B base eval (1109350) 완료 |
| **오늘 12:00~14:00** | Evol random_10 8 bench (1109879) 완료 |
| **오늘 14:00~15:00** | 14B random_10 eval 완료 |
| **5/22 새벽** | 14B tads_10 학습 완료 → 즉시 eval launch |
| **5/22 점심** | 14B tads_10 eval 완료 → scaling 표 채움 |
| **5/22 오후~저녁** | Evol tads_10 학습 완료 (1109881) → eval launch |
| **5/22 밤** | Evol tads_10 eval 완료 → Evol appendix 채움 |
| **5/23** | paper 마무리 + 제출 |
