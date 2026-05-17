# Experiment Plan — `thm_verify_05b` (Theorem 1 / App. F)

> **Audience:** human collaborators + AI assistants who need to understand,
> critique, or extend the empirical verification of Theorem 1 (Anchor
> Stability) for the CIKM 2026 TADS paper. The document is self-contained:
> no need to read the codebase to follow it.

---

## 0. TL;DR

- **What:** Run a single GPU 0.5B SFT pass that triggers our `TheoremVerifier`
  every 50 optimizer steps, records per-layer measurements of Theorem 1's
  assumptions and conclusion, and proves the theorem holds **empirically**
  on real LLM training (matching §4.5 of the paper).
- **Where:** SPACE cluster, node `run1108146-tads-theory`, GPU 0
  (A100-80GB, the other 7 GPUs idle and reserved for later runs).
- **Code:** `/group-volume/jieuns/tads_v2/` (remote
  `github.com/oliviajieuns/tads`, branch `main`).
- **Status:** Running. Anchor extraction phase observed; SFT loop entering.
- **Output root:**
  `/group-volume/jieuns/tads-checkpoints/light/thm_verify_05b/runs/<tag>/`.

---

## 1. Background — what Theorem 1 says

Trajectory-anchored selection score:

```
s_i  =  R_i · a_i · (1 + λ · align_i)
```

where `align_i = <h_i, v_l^(t)>` and `v_l^(t)` is the top-1 PCA direction of
per-layer hidden-state deltas at refresh point `t`. Theorem 1 asserts that
the **anchor direction is stable across refresh points**, formalised as a
two-part claim:

- **Conclusion C1 (per-step bound):**
  `‖v_l^(t+1) − v_l^(t)‖ ≤ (2 · C_Σ / γ_l) · η_t`
- **Conclusion C2 (cumulative bound):**
  `S_l(t) := Σ_{s≥t} ‖v_l^(s+1) − v_l^(s)‖  ≤  (2 C_Σ / γ_l) · Σ_{s≥t} η_s`

It relies on three assumptions:

| Tag | Assumption                                  | Operational form                |
|-----|---------------------------------------------|---------------------------------|
| A1  | Covariance Σ_l drifts smoothly with η       | `‖Σ^(t+1) − Σ^(t)‖_F ≤ C_Σ·η_t` |
| A2  | Eigengap γ_l > 0 throughout training        | `γ_l^(t) = λ1 − λ2 > 0`         |
| A3  | Sign calibration aligned across refreshes   | `sign(<v^(t+1), v^(t)>) ≥ 0`    |

This experiment instruments every term above and dumps them to disk.

---

## 2. Experiment configuration

### 2.1 Model & data

| Field             | Value                                                            |
|-------------------|------------------------------------------------------------------|
| Base model        | Qwen2.5-0.5B (24 decoder layers, hidden 896)                     |
| Local path        | `/group-volume/jieuns/models/Qwen2.5-0.5B`                       |
| Tuning mode       | LoRA (r=8, α=16, dropout=0.05; targets default LoRA module set)  |
| Dataset           | Alpaca-GPT4 (`/group-volume/IT-datasets/alpaca_gpt4/data/`)      |
| Tokenisation      | `qwen_chatml`, max_seq_len 512                                   |
| Selection ratio   | 0.1 (≈5,200 samples / epoch after TADS scoring)                  |
| Epochs            | 3                                                                |
| Batch size        | 4 micro × 2 grad_accum → effective 8                             |
| Optimizer         | torch.AdamW fp32, wd=0.1                                         |
| Selector          | RL composite reward + trajectory-anchor multiplier (λ from cfg)  |

Method: `tads` (i.e. λ > 0). For ablation comparisons, λ=0 reduces to the
RL+CR baseline; runs not part of this plan.

### 2.2 Verifier knobs (`configs/experiments/thm_verify_05b.yaml`)

```yaml
verification:
  enabled: true
  refresh_every_optstep: 50
  save_anchors: true
  track_delta_sigma: true
  probe_size: 256          # samples used for per-layer PCA at every refresh
  probe_seed: 4242         # FIXED across all refresh points → Σ drift
                           # reflects model drift only, not probe resampling
  output_subdir: thm_verification
  record_baseline: true    # extra t=0 measurement so the first row has finite
                           # baseline rather than NaN d / NaN ΔΣ
anchor:
  layer_indices: all       # per-layer Theorem verification needs every layer
  max_samples_for_pca: 256
  pca_batch_size: 4
```

### 2.3 Anchor refresh count (sanity)

```
|D| · selection_ratio ≈ 52,000 · 0.1 = 5,200 samples / epoch
effective_batch       = 4 · 2 = 8
opt_steps / epoch     ≈ 5,200 / 8 = 650
total opt_steps (3 ep)= 1,950
refresh count         ≈ 1,950 / 50  ≈ 39 measurement points
```

→ ≥ 20 measurement points (the lower bound for E2's linear regression).

---

## 3. What gets recorded

Every refresh writes one row per layer to
`<run_dir>/thm_verification/metrics.jsonl`:

```json
{
  "global_step": 50,
  "epoch": 1,
  "lr": 1.97e-5,
  "layer_idx": 11,
  "lambda1": 1.27e-2,
  "lambda2": 6.81e-3,
  "gamma": 5.89e-3,
  "delta_sigma_fro": 4.20e-4,
  "sign_inner_prev": 0.998,
  "d_step": 5.10e-5
}
```

Each refresh also dumps anchor vectors as
`anchors/step_<step:08d>.npy` of shape `(n_layers, hidden_dim)`, and once
writes `anchors/layer_indices.json` (the canonical layer ordering).

The verifier is rank-0 only; DDP workers see it as a no-op.

---

## 4. Offline analysis pipeline

`scripts/analyze_theorem.py` consumes the JSONL + `.npy` files and produces:

| Experiment | Question                                  | Output                              |
|-----------:|-------------------------------------------|-------------------------------------|
| **E1**     | Is γ_l > 0 everywhere? (A2)               | `F1_eigengap_heatmap.{png,pdf}`     |
| **E2**     | Fit `‖ΔΣ‖_F = C_Σ · η`. (A1)              | `F2_dSigma_vs_lr.{png,pdf}` + C_Σ   |
| **E3**     | Does `d ≤ (2 C_Σ / γ_min) · η_t` hold? (C1) | `F3_d_vs_bound.{png,pdf}`         |
| **E3b**    | Sign-flip count (A3 sanity)               | counter in `summary.json`           |
| **E4**     | Cumulative tail `S_l(t)` (C2)             | `F4_cumulative_S.{png,pdf}`         |
| **F1**     | Tightness ratio by layer group           | `table_F1_tightness.{csv,tex}`      |

The analyser also writes `summary.json` (machine-readable numbers for §4.5)
and `PASS_FAIL.json` with explicit verdicts:

| Verdict | Pass condition                                                                                  |
|---------|-------------------------------------------------------------------------------------------------|
| E1      | `γ_min > 0` **and** zero γ ≤ 0 violations                                                       |
| E2      | finite C_Σ with `R² > 0.3`, n > 0                                                               |
| E3      | zero bound violations **and** mean tightness ratio ∈ (0.001, 1.0) (`VACUOUS` if ≤ 0.001)        |
| E3b     | zero sign flips                                                                                 |
| E4      | ≥ 80% of layers have monotone non-increasing `S_l(t)`                                           |

Tightness ratio = `d^(t) / bound^(t)`. ≪ 1 means the bound is loose
(slack); >1 is a violation.

---

## 5. Execution

### 5.1 Launch (already running)

```bash
cd /group-volume/jieuns/tads_v2
source /group-volume/jieuns/llm-instruction-tuning/venv/bin/activate
tmux new -d -s tads_05b
tmux send-keys -t tads_05b \
  "CUDA_VISIBLE_DEVICES=0 python -m tads.train \
    --config /group-volume/jieuns/tads_v2/configs/experiments/thm_verify_05b.yaml \
    2>&1 | tee /group-volume/jieuns/tads_v2/logs/thm_verify_05b_run1108146.log" C-m
```

### 5.2 Health checks during run

```bash
RUN=/group-volume/jieuns/tads-checkpoints/light/thm_verify_05b/runs/<tag>

# refresh cadence
wc -l $RUN/thm_verification/metrics.jsonl
ls $RUN/thm_verification/anchors/ | head

# GPU
nvidia-smi -i 0
```

Expected progression: ~13 refreshes / epoch → JSONL grows by ~13 × L lines
(L = number of decoder layers = 24 for Qwen2.5-0.5B) every epoch.

### 5.3 Post-run analysis

```bash
python scripts/analyze_theorem.py --run_dir $RUN
cat $RUN/thm_verification/analysis/PASS_FAIL.json
```

All five verdicts must be `PASS` (or `VACUOUS` for E3 in the extreme low-LR
case) before drafting the App. F numbers into the paper.

---

## 6. Paper-side deliverables (App. F + §4.5)

| Paper element                                | Source artifact                                          |
|----------------------------------------------|-----------------------------------------------------------|
| §4.5 narrative numbers (γ_min, C_Σ, mean ratio) | `summary.json`                                         |
| Figure 1 (eigengap heatmap)                  | `F1_eigengap_heatmap.pdf`                                 |
| Figure 2 (Σ-drift vs lr)                     | `F2_dSigma_vs_lr.pdf`                                     |
| Figure 3 (per-step bound)                    | `F3_d_vs_bound.pdf`                                       |
| Figure 4 (cumulative tail)                   | `F4_cumulative_S.pdf`                                     |
| Table F1 (tightness by layer group)          | `table_F1_tightness.tex` (booktabs-ready)                 |

`summary.json` also carries the `verdicts` dict, which lets the LaTeX
template auto-stamp a “verified on N=… measurement points, R²=…” caption.

---

## 7. Failure modes and how the experiment exposes them

| Failure                                       | Symptom                       | Surfaces as           |
|-----------------------------------------------|-------------------------------|-----------------------|
| Eigengap collapses (A2 broken)                | `gamma ≤ 0` row in JSONL      | E1 = FAIL             |
| Σ drift super-linear in η (A1 broken)         | R² ≪ 1 in F2 scatter          | E2 = FAIL             |
| Anchor flips sign across refreshes (A3)       | `sign_inner_prev < 0` row     | E3b = FAIL            |
| C1 bound exceeded (theorem false)             | `d > bound`                   | E3 = FAIL             |
| C2 tail non-monotone (theorem false)          | `S_l(t+1) > S_l(t)`           | E4 = FAIL             |
| Probe resampling contaminates Σ measurement   | ΔΣ noisy regardless of η      | E2 = FAIL (false-negative) |

The fixed probe (point 2.2) + step-level cadence + per-layer reporting are
designed so that any failure on the left maps to exactly one verdict on the
right, with no aliasing across causes.

---

## 8. Scope / non-goals

- **Not** comparing TADS against baselines. That is a separate experiment
  set (`evol_7b`, NAIT 9-task harness).
- **Not** measuring downstream task quality. App. F is a *theorem
  verification*, not an ablation.
- **Not** a scaling study. 0.5B was chosen because (i) the theorem is
  scale-agnostic, (ii) per-step PCA at every refresh × 24 layers must fit
  in a single A100. Validating on 7B/14B is future work and lands in App. G.

---

## 9. Reproducibility checklist

- [x] Fixed probe seed (`probe_seed: 4242`) — Σ drift is model-only.
- [x] Anchors dumped per refresh as `.npy` for offline re-analysis.
- [x] Analyser deterministic (no random state).
- [x] PASS/FAIL JSON committed alongside figures.
- [x] Config + code on `main` (commit `f235121` or later).
- [ ] Final run hash + log added to paper artefact appendix once finished.

---

## 10. Pointers

| Asset                              | Path                                                      |
|------------------------------------|-----------------------------------------------------------|
| Train entrypoint                   | `tads/train.py`                                           |
| Verifier                           | `tads/core/thm_verification.py`                           |
| Anchor (track_sigma path)          | `tads/core/trajectory_anchor.py`                          |
| SFT hook                           | `tads/pipelines/sft.py` (one line at optimizer boundary)  |
| Config                             | `configs/experiments/thm_verify_05b.yaml`                 |
| Analyser                           | `scripts/analyze_theorem.py`                              |
| Tests                              | `tests/test_theorem_verification.py`                      |
| Project memory (paths, rules)      | `CLAUDE.md`                                               |
