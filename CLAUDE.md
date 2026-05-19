# TADS — Claude 작업 메모리

## 프로젝트 개요

TADS (Trajectory-Anchored Dynamic Data Selection) — CIKM 2026 Full Paper.
LLM instruction tuning을 위한 동적 데이터 선택 프레임워크. RL composite-reward
base selector에 모델 자신의 hidden-state trajectory에서 뽑은 구조적 anchor를
곱셈 항으로 주입. Theorem 1 (Anchor Stability)이 main contribution.

- 제출: CIKM 2026, 5/23 마감. Rome.
- 코드: /user-volume/tads/ , GitHub github.sec.samsung.net/jieun/tads
- 핵심 수식: s_i = R_i · a_i · (1 + λ · align_i), λ=0이면 base selector

---

## LLM 실험 현황

- 모델: 0.5B ~ 14B 멀티스케일
  - Qwen2.5-0.5B (sanity track / 빠른 검증·디버그)
  - LLaMA-2-7B (메인 실험, NAIT 프로토콜)
  - Qwen2.5-14B (scaling study)
- 학습 데이터: Alpaca-GPT4 + Evol-Instruct 계열 + 그 외 1종 (멀티 데이터셋)
- 메인 실험: LLaMA-2-7B Full fine-tuning, Alpaca-GPT4, 10% 선택
  (NAIT ICLR 2026 프로토콜 그대로 — 직접 비교 가능하게)
- 14B scaling study: LoRA SFT (r=16, alpha=32,
  target=q/k/v/o/gate/up/down_proj)
- 평가: 다중 벤치마크 (NAIT 9-task 체계)
  - 지식: MMLU, MMLU-Pro
  - 수학 추론: GSM8K, SVAMP
  - 코드: HumanEval, MBPP
  - 다국어: TydiQA, XQuAD
  - 일반 추론: BBH
- 평가 도구: lm-evaluation-harness
- baseline: Random, Full, LIMA, AlpaGasus, Q2Q, SelectIT, NAIT,
  RL+CR (= λ=0 instance)

---

## SPACE 클러스터 환경

- run 번호가 다르면 완전히 다른 환경 (group-volume, venv, 모델 모두 분리)
- user-volume만 동일 사용자 기준 모든 노드 공유
- group-volume은 노드마다 별개 mount일 수 있음 — 노드별 검색 따로
- 터미널 프롬프트의 run{번호} 확인 → 즉시 해당 노드 기준으로 경로 판단

경로:
- venv: /group-volume/jieuns/llm-instruction-tuning/venv/bin/python
- 0.5B: /group-volume/jieuns/models/Qwen2.5-0.5B
- 14B:  /group-volume/jieuns/models/Qwen2.5-14B
- LLaMA-2-7B: (경로 미확정 — 사용 시 확인)
- 학습 데이터: /group-volume/IT-datasets/alpaca_gpt4/data/
- 코드 (사내 클론): /user-volume/tads/ (github.sec.samsung.net/jieun/tads)
- 코드 (실제 import 경로, fork 클론): /group-volume/jieuns/tads_v2/
  remote = github.com/oliviajieuns/tads (메인 작업 repo)
  venv에 editable 설치됨: `pip install -e /group-volume/jieuns/tads_v2`
- 백업: /group-volume/jieuns/tads_v2.bak.<YYYYMMDD_HHMM>
  (구 33modeling/tads 클론 + WIP evol_7b configs + run_evol_chain_after_random.sh)
- 출력/체크포인트: /group-volume/jieuns/tads-checkpoints/
  (configs/base.yaml의 output_root + data_cache 둘 다 본인 경로로 수정 완료;
   `${oc.env:OUTPUT_ROOT,...}` / `${oc.env:DATA_CACHE,...}` 기본값)
  ⚠️ 동료 minsoo3.kim 경로 박힌 config 발견 시 본인 경로로 교체 필수

노드별 메모:
- run1108146-tads-theory: A100-SXM4-80GB × 8 (sanity/0.5B 작업 노드)

### 2026-05-19 노드 (오늘 세션 임시 매핑, 1 GPU per node)

| 노드 | GPU | MIG | 가용 | 용도 |
|---|---|---|---|---|
| run1108843-tads2-jieun-pai | A100-80GB | OFF | 80GB full | **7B Qwen2.5 메인 #1 (LoRA)** |
| run1108842-tads-jieun-pai  | A100-80GB | OFF | 80GB full | **7B Qwen2.5 메인 #2 (LoRA)** |
| run1107736-tads-a100-g20   | A100-80GB | ON  | 40GB (3g.40gb) | GSM8K eval 진행중 → 끝나면 0.5B |
| run1107739-tads1           | A100-80GB | ON  | 40GB | 0.5B sweep |
| run1108824-tads2-jieun     | A100-80GB | ON  | 40GB | 0.5B sweep |

⚠️ 노드 간 ssh 안 됨 (hostname resolution 실패). 각 노드에서 직접 launch.
⚠️ 1 GPU per node → 7B full FT 어려움 → LoRA 로 가는 게 현실적.

---

## 코드 제공 규칙 (항상 준수)

- 항상 코드블록으로 제공
- 실험 실행 코드 제공 시 반드시 sleep 60~90 && tail 로그 확인 명령 함께
- 절대 경로만 사용
- 항상 한국어로 답변
- python/pip 명령 제안 전 터미널 프롬프트에 (venv) 표시 확인.
  없으면 먼저: source /group-volume/jieuns/llm-instruction-tuning/venv/bin/activate

---

## ⚠️ 안전 규칙 (절대 금지)

- pkill -9 python / killall python / fuser -k /dev/nvidia* 절대 금지
  → jupyter 노트북 노드까지 죽임 (실제 사고 이력)
- 안전한 대안: pkill -9 -f "lm_eval --model" (specific) /
  kill -9 <PID> (정확한 PID)
- generic python kill 명령 어떤 상황에서도 제안 금지

---

## 답변 스타일

- 한 답변에 너무 많은 옵션/긴 설명 나열 금지
- 짧고 토큰 절약하되 현재 액션 + 다음 1~2 step 미리보기
- 긴 timeline·여러 분기 시나리오·모든 옵션 나열 금지
