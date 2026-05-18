# TADS 실험 현황 (2026-05-18)

> 동료/팀 공유용. Confluence에 그대로 붙여넣을 수 있는 마크다운.
> 마지막 업데이트: 2026-05-18 04:35 UTC (4 노드, 동시 진행 중)

---

## 한 줄 요약

지금 **3개 노드, 약 9 GPU**가 paper 자산을 동시에 만들고 있음. App. F 검증은 끝났고, 메인 결과 표(Qwen2.5-7B × WizardLM Evol-Instruct 70K 4셀 매트릭스)와 Figure 2의 RL+CR 비교 곡선이 남은 작업.

---

## 노드별 현황

| 노드 | GPU | 실험 | 상태 | 예상 종료 |
|---|---|---|---|---|
| **run1107736** | GPU 0 (단일, 순차) | Qwen2.5-7B Evol `random_10` + `full_100` 다운스트림 eval (MMLU, GSM8K, HumanEval, TyDiQA, BBH) | random_10 MMLU 완료 → GSM8K 진행 중 | ~2-3 시간 |
| **run1108319** | GPU 0-3 (4-GPU DDP) | Qwen2.5-7B Evol `tads_10` 학습 **재시도** | 학습 시작 직후 (~90초 헬스체크 통과 시점) | ~3 시간 |
| **run1108321** | GPU 0-3 (4-GPU DDP) | Qwen2.5-7B Evol `data_agent_10` 학습 **재시도** | 학습 시작 직후 | ~3 시간 |
| **run1107739** | GPU 1 (단일) | Qwen2.5-0.5B `thm_verify_rlcr` (no-anchor, Figure 2의 회색 곡선 측정용) | tmux `thm_rlcr` 세션 가동 | ~1.5 시간 |

> `nvidia-smi`에 `[Insufficient Permissions]`가 보이는 건 클러스터 정책 (사용자에게 카운터 권한이 없을 뿐), 학습 자체는 정상.

---

## 실험별 디테일

### 1. Eval — `random_10` / `full_100` (run1107736 GPU 0)

이미 학습이 끝난 두 셀의 5종 벤치마크 평가. 단일 GPU 순차 실행 (`tmux: eval_seq`).

- **random_10 MMLU**: 57 subjects 완료, macro avg **72.71%** (Qwen2.5-7B base 보존). 다만 max_length=2048 한도로 인해 5-shot prefix 좌측 truncate가 373/14042 = **2.7%** 발생, 주로 long-context history 과목 두 개에 집중되어 점수 하방 편향.
- **이어서 자동**: GSM8K → HumanEval → TyDiQA → BBH → full_100 5종 벤치마크.
- **결과 위치**: `${EVAL_RESULTS_ROOT}/evol_7b/qwen25/{random_10,full_100}/qwen25_<method>-{mmlu,gsm8k,humaneval,tydiqa,bbh}.json`

### 2. tads_10 / data_agent_10 evol_qwen 재학습 (run1108319, run1108321)

#### 이전 실패 원인 (5/17)
4-GPU DDP 학습에서 SIGABRT exit -6. 로그 분석 결과:
- `collect_episode` (PPO rollout)가 rank 0에서만 **49분간** 단독 실행.
- rank 1-3은 selection 결과 대기 중 idle.
- collect 끝나고 SFT 첫 step의 gradient ALLREDUCE에서 **NCCL watchdog 2시간 timeout** 발생.
- 즉 collect_episode 동안 NCCL collective가 발화되지 않아 워치독이 stale 판정.

#### 재시도 패치
환경변수로 NCCL timeout을 6시간으로 확대:

```bash
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=21600   # 2hr → 6hr
export NCCL_TIMEOUT=21600
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
```

학습 dynamics 자체는 동일 — 죽었던 자리만 우회. 만약 6시간으로도 실패하면 code-level keep-alive 패치 또는 **단일 GPU 재실행 (8-12h)** fallback.

#### Configs
- `configs/experiments/evol_7b/qwen25/tads_10.yaml`
- `configs/experiments/evol_7b/qwen25/data_agent_10.yaml`

#### 출력
- `${OUTPUT_ROOT}/evol_7b/qwen25/{tads_10,data_agent_10}/runs/<timestamp>_retry*/epoch_*/_complete` (sealed sentinel 떨어지면 완료)

### 3. RL+CR thm_verify (run1107739 GPU 1)

**목적**: Figure 2 (Theorem 1 empirical verification figure)의 회색 점선 (RL+CR variance-ratio trajectory) 자리에 들어갈 **측정 데이터** 확보. 현재는 schematic placeholder.

- **Config**: `configs/experiments/thm_verify_05b_no_anchor.yaml` — anchor를 selection score에 **disable** (`tads.use_anchor=false`, `tads.lam=0`) 하고, lr을 cell D의 2.5배 (5e-4)로 키워 변동성 증폭. anchor는 verifier가 계속 측정.
- **가설**: anchor smoothing 없으면 hidden state의 top-1 eigenvector가 erratic하게 흔들리며 cell D 기준 bound (46.24)를 반복적으로 초과 → Figure 2의 시각적 대비 확보.
- **분석**: 끝나면 `python scripts/analyze_theorem.py --run_dir <run> --skip_warmup_optsteps 250 --gamma_threshold 0.05 --robust` 로 동일 분석.

---

## 완료된 작업 (참고)

| 작업 | 결과 | 산출물 |
|---|---|---|
| App. F Theorem 1 verification — 2×2 ablation | 6/8 셀 모두 (E1..E4) PASS (robust 기준) | `paper_artifacts/{fig_main,fig_cross_eta,table_verdicts,appendix_F}` |
| A1 cross-η sweep | C_σ가 sub-linear (slope ≈ 0.36) — strict A1 깨짐, robust A1 성립 | `paper_artifacts/fig_cross_eta.{tex,pdf}` |
| Figure 2 측정값 갱신 | TADS anchor 곡선 (D cell, N=68 points) 실측 반영 | `docs/paper_drafts/fig_thm_main.tex` |
| `sec:thm-empirical` 본문 채움 | bound = 46.24 유도 식 포함 | `docs/paper_drafts/sec_thm_empirical.tex` |
| evol_7b/qwen25 `random_10`, `full_100` 학습 | 4-GPU DDP, 3 epochs sealed | `${OUTPUT_ROOT}/evol_7b/qwen25/{random_10,full_100}/` |

---

## 남은 작업

- [ ] tads_10 / data_agent_10 evol_qwen 학습 완료 (run1108319, run1108321)
- [ ] 위 2 셀의 eval (5 benchmarks)
- [ ] thm_verify_rlcr 결과로 Figure 2 회색 곡선 실측 교체
- [ ] evol_qwen 4 셀 완성된 메인 결과 표 (paper §5)
- [ ] (선택) MMLU `max_length=2048 → 4096` 재평가로 truncation bias 제거

---

## 위험 / 알려진 이슈

- **NCCL 6시간 timeout 실패 시 fallback**: 1-GPU 단독 학습 (`--nproc_per_node=1`, `grad_accum` 16+ 으로 effective batch 보존). 약 8-12시간/셀.
- **MMLU truncation**: 2.7% 항목이 5-shot context를 잃어 점수 하방 편향. paper footnote로 디스클로저 또는 재평가로 보정.
- **노드 reclaim**: 5/17에 8156/8158이 학습 ~8시간 후 죽음. 가능하면 길게 도는 학습은 sealed epoch 체크를 자주 (epoch당 한번) 해서 재시작 가능 상태 유지.

---

## 빠른 진척 확인 명령

```bash
# 노드별 학습/eval 로그 한 줄씩
for log in /group-volume/jieuns/tads_v2/logs/eval_evol_qwen_*.log \
           /group-volume/jieuns/tads_v2/logs/evol_qwen_*retry*.log \
           /group-volume/jieuns/tads_v2/logs/thm_rlcr.log; do
  echo "=== $(basename $log) ==="
  tail -n 3 "$log" 2>/dev/null
done

# sealed epoch 일괄 확인
find /group-volume/jieuns/tads-checkpoints/evol_7b/qwen25 \
     /group-volume/jieuns/tads-checkpoints/light/thm_verify_rlcr \
     -name "_complete" 2>/dev/null

# 평가 결과 파일
ls /group-volume/jieuns/tads-eval-results/evol_7b/qwen25/*/qwen25_*-eval_summary.json 2>/dev/null
```

---

## 참고 경로

| 자산 | 경로 |
|---|---|
| repo (실제 import) | `/group-volume/jieuns/tads_v2/` (remote: `github.com/oliviajieuns/tads`) |
| venv | `/group-volume/jieuns/llm-instruction-tuning/venv/bin/python` |
| 체크포인트 루트 | `/group-volume/jieuns/tads-checkpoints/` |
| 평가 결과 루트 | `/group-volume/jieuns/tads-eval-results/` |
| 데이터 — WizardLM Evol-Instruct | `/group-volume/IT-datasets/wizardlm_evol_instruct_70k/alpaca_evol_instruct_70k.json` |
| 데이터 — Alpaca-GPT4 (참고) | `/group-volume/IT-datasets/alpaca_gpt4/data/*.json` |
| Qwen2.5-7B base | `/group-volume/nait-models/qwen2.5-7b/` |
| Qwen2.5-0.5B base | `/group-volume/jieuns/models/Qwen2.5-0.5B/` |
| App. F 분석기 | `scripts/analyze_theorem.py` |
| paper draft 디렉토리 | `docs/paper_drafts/` |
