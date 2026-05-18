# TADS Additional Experiments — CIKM 2026 마무리 계획

> 마감 (5/23) 전 reviewer 대응까지 염두에 둔 추가 실험 plan.
> 두 축 (model scale, hyperparameter sensitivity) + 그 외 must/nice-to-have.

---

## 우선순위 요약

| 우선순위 | 실험 | 이유 | 예상 compute |
|---|---|---|---|
| **P0 (필수)** | 14B scaling | "Scale robustness" — reviewer 최우선 질문 | ~24 GPU-h |
| **P0** | λ-ablation (0, 0.5, 1, 2, 5, 10) | hyperparameter sensitivity — TADS 고유 knob 정당화 | ~18 GPU-h |
| **P0** | selection ratio sweep (5%, 20%, 50%) | 10% 선택의 합리성 + 일반화 | ~12 GPU-h |
| **P1** | wall-clock / compute overhead 표 | 비용-효익 분석 (TADS의 PCA / rollout 오버헤드) | ~0 (기존 로그 파싱) |
| **P1** | refresh interval ablation | App. F 가정 robustness | ~6 GPU-h (0.5B) |
| **P1** | anchor variants (`layer_indices`) | 우리만의 design choice 정당화 | ~9 GPU-h |
| **P2** | cross-dataset generalization | Alpaca↔Evol train/test swap | ~6 GPU-h |
| **P2** | loss curve / 수렴 분석 | 기존 로그 파싱 + 그림 | ~0 |
| **P2** | qualitative example analysis | "what does TADS select differently" | ~2 GPU-h |

---

## P0-1. 14B scaling study

**설정**: Qwen2.5-14B + LoRA (메모리 절약 — 14B full FT는 4×A100 80GB 빠듯)
- LoRA: `r=16, alpha=32, target=q/k/v/o/gate/up/down_proj` (CLAUDE.md 기준)
- 4 cell: `random_10`, `data_agent_10`, `tads_10`, `full_100`
- 같은 5 benchmark (mmlu, gsm8k, humaneval, tydiqa, bbh)

**configs to create**: `configs/experiments/main_14b/qwen25/{random,data_agent,tads,full}_*.yaml`
**예상 GPU**: 4-GPU DDP × 4 cells × ~6h = ~24h
**메시지**: "TADS의 효과가 모델 크기에 따라 증폭/약화되는가?" — 통상 reviewer는 TADS 마진이 큰 모델에서도 유지되는지 본다.

---

## P0-2. λ-ablation (앵커 강도 sweep)

**설정**: Qwen2.5-7B + LoRA + Alpaca-GPT4 또는 Evol (1 dataset 고정), 10% selection
- λ ∈ {**0, 0.5, 1, 2, 5, 10**} — 6 cells
- λ=0이면 정확히 `data_agent`와 동등 (이미 있음, 재사용 OK)
- λ=1이 메인 표의 `tads_10` (이미 있음)
- 따라서 **새로 돌릴 cell은 λ ∈ {0.5, 2, 5, 10} = 4개**

**override 방식 (config 파일 안 만들고)**:
```bash
python -m tads.train --config configs/experiments/main_7b/qwen25/tads_10.yaml \
    --override tads.lam=2.0 --run_suffix=lam2
```

**예상 GPU**: 4-GPU DDP × 4 새 cells × ~3.5h = ~14h, plus eval ~4h → 총 ~18h
**메시지**: "TADS의 λ=1 선택이 최적인가? 너무 작거나 너무 크면 망가지는가?" — 강한 ablation.

**기대 모양**: U-shape — 너무 작으면 anchor 효과 약함, 너무 크면 다른 신호 묻힘. λ=1 근방에서 최댓값.

---

## P0-3. Selection ratio sweep

**설정**: Qwen2.5-7B + LoRA + Alpaca-GPT4 (또는 Evol), TADS, λ=1
- ratio ∈ {**0.05, 0.1, 0.2, 0.5**} — 4 cells
- ratio=0.1은 이미 있음 → 새로 돌릴 건 3개
- ratio=1.0 은 `full_100` (이미 있음)

**override**:
```bash
python -m tads.train --config configs/experiments/main_7b/qwen25/tads_10.yaml \
    --override selection_ratio=0.2 --run_suffix=ratio20
```

**예상 GPU**: ratio가 작으면 epoch당 학습 step 적어서 빠름. 4-GPU DDP × 3 cells × ~3-4h = ~12h
**메시지**: "10% 선택이 sweet spot인가, 더 적게 / 더 많이 해도 효과 유지되나?" — TADS의 'data efficiency' 주장 정당화.

**기대 모양**: ratio 작을수록 TADS의 anchor selection 효과가 극대화 (random은 망함). ratio 50%에 가면 차이 감소.

---

## P1-1. Wall-clock / compute overhead 표

**설정**: 기존 logs를 grep해서 구성. 새 학습 불필요.
- 각 메서드의 collect_episode 시간, anchor PCA 시간, SFT step 시간 합산
- TADS = data_agent + (anchor PCA overhead)
- random = SFT only

**산출**: paper Table — 메서드별 wall-clock breakdown.
**메시지**: "TADS가 정확도 이득에 비해 어느 정도 overhead가 있나?"

---

## P1-2. Refresh interval ablation (App. F 보강)

**설정**: thm_verify_05b.yaml 기반으로 refresh_every_optstep ∈ {10, 25, 50, 100}
- 0.5B + LoRA 단일 GPU × 4 cells × ~1.5h = ~6h
- 우리의 default (25)가 합리적인지 보여주는 robustness 분석
- bound 값과 tightness 변화 추이

**메시지**: "refresh를 너무 자주 / 드물게 해도 anchor 안정성이 깨지지 않나?" — App. F만 보강하고 본문 영향 없음.

---

## P1-3. Anchor variants (`layer_indices` choice)

**설정**: 같은 데이터/모델로 layer 선택만 다르게
- `all` (default), `middle_to_last` (legacy), 단일 layer `-1`
- 3 cells × ~3.5h = ~10h
**메시지**: "왜 all layers 사용하나? 한 layer만 하면 충분하지 않나?" — design choice 정당화.

---

## P2-1. Cross-dataset generalization

**설정**: Alpaca로 학습 → Evol 스타일 eval (또는 vice versa)
- TADS의 selection이 dataset-specific 아니라는 증거
- 4 cell (기존 random/tads/full × 2 데이터셋)이미 있음 → 새 학습 없이 eval만

---

## P2-2. Loss curve / 수렴 분석

**기존 학습 log에서 step별 loss 추출** → matplotlib plot.
**메시지**: "TADS가 더 빠르게 수렴하는가?" 학습 정성 분석.

---

## P2-3. Qualitative example analysis

**설정**: TADS가 선택한 sample vs data_agent가 선택한 sample의 차이
- `selected_indices_epoch{N}.json` 비교
- 예시 5-10개 paper appendix에 (TADS preferred, both selected, neither selected)
**메시지**: anchor가 어떤 종류의 sample을 더 골라내는지 직관 제공.

---

## 추천 실행 순서 (마감 역산)

| 시점 | 실행 |
|---|---|
| **지금~+24h** | (현재 진행 중) evol_qwen 4 cell + eval + RL+CR thm_verify |
| **+24h ~ +48h** | P0-2 λ-ablation 4 cells (병렬 4-GPU DDP × 2 노드면 ~14h) |
| **+48h ~ +96h** | P0-1 14B scaling 4 cells + eval (~24h 학습 + eval) |
| **+96h ~ +120h** | P0-3 selection ratio sweep 3 cells (~12h) |
| **마감 -24h** | P1-1 wall-clock 표 (paper 본문) + P1-2/P1-3 가능하면 |

각 실험은 **상황에 따라 1-GPU fallback** 가능하지만 ~3-4배 시간 소요.

---

## 즉시 실행 가능한 것들 (compute 불필요)

- **P1-1 wall-clock 표** — 기존 log 파싱
- **P2-2 loss curve** — 기존 metrics.json 파싱
- **P2-3 qualitative sample** — 기존 selected_indices*.json 비교

마감 직전에 자투리 시간으로 이것들 한 번에 처리.

---

## 결정해야 할 사항

1. **14B 모델 경로 확인**: `/group-volume/jieuns/models/Qwen2.5-14B` 존재 확인 필요
2. **14B dataset**: alpaca-gpt4 (메인 매트릭스 연속) vs evol-instruct (dataset-axis 연속)
3. **λ-ablation dataset**: alpaca-gpt4 1개로 충분 vs evol에서도 해야 하는지
4. **새 configs를 PR로 만들지** (재사용성 ↑) vs `--override`로 즉시 launch (빠름)

위 4개 선호도 알려주면 P0 3개 실험의 config + launcher 한꺼번에 PR로 올릴게.
