# Experiment Plan — `evol_7b/qwen25` (Qwen2.5-7B × WizardLM Evol-Instruct 70K)

> **Audience:** human collaborators + AI assistants. Self-contained: every
> hyperparameter, path, and policy is spelled out so this experiment can be
> reproduced, reasoned about, or extended without reading the codebase.

---

## 0. TL;DR

- **Question (single-axis ablation):** Does TADS' relative ranking against
  baselines (`random_10`, `data_agent_10`, `full_100`) hold when the training
  data is **WizardLM Evol-Instruct 70K** instead of **Alpaca-GPT4 52K**?
  Model, selector, eval pipeline, and hyperparameters are all held constant.
- **Model:** Qwen2.5-7B (full FT, DDP × 4 GPUs)
- **Methods:** `random_10`, `data_agent_10`, `tads_10`, `full_100` (4 cells)
- **GPU layout (this run):** 4-GPU DDP per cell, **2 parallel chains** =
  GPU 0-3 (chain A) + GPU 4-7 (chain B). Stays at 100% utilisation so the
  cluster policy does not reclaim the node.
- **Wall time estimate:** ≤ 8 h end-to-end (chain B bottlenecked by
  `full_100`, ~3-4 h).
- **Counterpart matrix:** `main_7b/qwen25/*` (same model, same methods, but
  Alpaca-GPT4 data). The two matrices are directly comparable cell-by-cell.

---

## 1. Background — why this experiment

The main paper matrix (`main_7b/<model>/<method>`) holds the **dataset
constant** (Alpaca-GPT4) and varies model + selection method. That tests
"does TADS work across models on a fixed dataset?". The evol-instruct
matrix does the **orthogonal** ablation: hold the model constant and vary
the dataset. Failure modes:

- If TADS' lead over `random_10` disappears on Evol-Instruct, the headline
  claim becomes "TADS works on Alpaca but not on harder instruction data" —
  a strict negative result worth knowing.
- If TADS' margin grows on Evol-Instruct, that's evidence that anchor-based
  selection helps **more** when the data distribution is heterogeneous
  (Evol-Instruct mixes longer / multi-step instructions than Alpaca).
- If everything tracks Alpaca within noise → the contribution is
  **dataset-agnostic**, which is the strongest claim.

Why **Qwen2.5-7B** specifically? It's the strongest 7B base in the matrix
(MMLU ≈ 71 zero-shot) and uses a known SFT recipe (Qwen2.5 Technical Report)
that we already encode in `configs/models/qwen2.5-7b.yaml`. Llama-2-7B
evol_7b results already exist from a previous WIP run (May 14), giving us
a **second model row "for free"** when both matrices are written up.

---

## 2. Data

| Field          | Value                                                                                  |
|----------------|----------------------------------------------------------------------------------------|
| Source         | WizardLM Evol-Instruct 70K (Xu et al., 2023)                                          |
| Local path     | `/group-volume/IT-datasets/wizardlm_evol_instruct_70k/alpaca_evol_instruct_70k.json`   |
| Schema         | `[{instruction, output}]` — Alpaca-format JSON; **no `input` field**                  |
| Sample count   | 70,000                                                                                 |
| Tokenisation   | `qwen_chatml` prompt style (from `configs/models/qwen2.5-7b.yaml`)                     |
| `max_seq_len`  | 512                                                                                    |
| Loader         | `tads/data/alpaca.py::build_alpaca_dataset` — reused as-is. The missing `input` falls back to `""` via `example.get("input", "")` in `sft_prompts.py`. |

The data swap is driven entirely by two YAML keys (no env vars, no code
change):

```yaml
data_files: /group-volume/IT-datasets/wizardlm_evol_instruct_70k/alpaca_evol_instruct_70k.json
dataset_name: WizardLMTeam/WizardLM_evol_instruct_70k
```

---

## 3. Model

| Field            | Value                                          |
|------------------|------------------------------------------------|
| Base             | Qwen2.5-7B                                     |
| Local path       | `/group-volume/nait-models/qwen2.5-7b/`        |
| Files            | `config.json`, `model-0000{1..4}-of-00004.safetensors`, `merges.txt`, `tokenizer.json`, ... |
| `model_key`      | `qwen2.5-7b` (env override: `MODEL_PATH_QWEN25_7B`) |
| Prompt style     | `qwen_chatml` (sets ChatML wrappers around instruction/response) |
| Training mode    | **full fine-tuning** (no LoRA)                 |
| Precision        | bf16 (HF default for Qwen2.5)                  |
| Attention impl   | `null` → HF picks (typically SDPA; FA2 if installed) |
| Grad checkpoint  | **on** (essential for 7B + DDP on 80GB)        |
| Optimizer        | **8-bit AdamW** (`use_8bit_optimizer: true` from `configs/modes/full_ft.yaml`) |

---

## 4. Training hyperparameters (resolved after `defaults:` chain)

All numbers below are what each cell actually trains with, after the
`base → method → model → mode → experiment` override chain.

### 4.1 Common to all four cells

| Field                | Value                  | Source                                  |
|----------------------|------------------------|-----------------------------------------|
| `seed`               | 42                     | `base.yaml`                             |
| `train_epochs`       | 3                      | `evol_7b/qwen25/*.yaml` (explicit)      |
| `batch_size` (per-GPU) | 8                    | `modes/full_ft.yaml`                    |
| `grad_accum`         | 4                      | `modes/full_ft.yaml`                    |
| **Effective batch**  | **8 × 4 × 4 GPUs = 128** | (per NAIT Table 8)                    |
| `learning_rate`      | **1.0e-5**             | `models/qwen2.5-7b.yaml` (Qwen2.5 SFT recipe — half of Llama) |
| `warmup_ratio`       | 0.03                   | `base.yaml`                             |
| `weight_decay`       | 0.1                    | `base.yaml`                             |
| `gradient_clip`      | 1.0                    | `base.yaml`                             |
| `gradient_checkpointing` | true               | `base.yaml`                             |
| `use_8bit_optimizer` | true                   | `modes/full_ft.yaml`                    |
| `max_seq_len`        | 512                    | `models/qwen2.5-7b.yaml`                |
| `prompt_style`       | `qwen_chatml`          | `models/qwen2.5-7b.yaml`                |

### 4.2 Per-cell overrides

| Cell             | `selection_ratio` | `method`     | `episode_batch_size` | `tads.lam` | `tads.use_anchor` | Notes |
|------------------|-------------------|--------------|----------------------|------------|--------------------|-------|
| `random_10`      | 0.1               | `random`     | 1 (unused)           | n/a        | n/a                | No PPO rollout; pure SFT after random sampling. Lower-bound baseline. |
| `data_agent_10`  | 0.1               | `data_agent` | 16                   | 0.0        | false              | PPO rollout active; anchor disabled. **TADS λ=0 ablation companion.** |
| `tads_10`        | 0.1               | `tads`       | 16                   | 1.0        | true               | PPO rollout + anchor. **Headline method.** |
| `full_100`       | 1.0               | `full`       | 1 (unused)           | n/a        | n/a                | No selection; SFT on all 70K samples × 3 epochs. Upper-bound baseline. |

### 4.3 PPO Agent (used by `data_agent_10`, `tads_10` only)

From `base.yaml: agent:` — identical across both methods.

```yaml
agent:
  lr: 3.0e-4
  clip_eps: 0.2
  gamma: 0.99
  gae_lam: 0.95
  ppo_epochs: 4
  entropy_coef: 0.01
  value_coef: 0.5
  mb_size: 1024
  advantage_mode: group_relative
  value_clip: true
```

### 4.4 Trajectory Anchor (used by `tads_10` only)

From `evol_7b/qwen25/tads_10.yaml` (explicit) + `base.yaml: anchor:`:

```yaml
anchor:
  layer_indices: all          # all 28 Qwen2.5-7B decoder layers contribute
  layer_idx: -1               # back-compat single-layer key (ignored when layer_indices=all)
  max_samples_for_pca: 1024   # paper Eq. 5; lower than base (2000) to bound PCA wall time
  pca_batch_size: 16
```

**Anchor pipeline:** forward 1024 randomly sampled examples through the
model, take hidden-state deltas at each decoder layer, run top-1 PCA per
layer, sign-calibrate against the previous refresh, store as `v_l^(t)`.
Used to multiply the RL composite reward by `(1 + λ · align)` during
candidate scoring.

---

## 5. Output layout (run-layout, history-preserving)

```
/group-volume/jieuns/tads-checkpoints/
└── evol_7b/
    └── qwen25/
        ├── random_10/
        │   ├── runs/<YYYYMMDD_HHMMSS>/
        │   │   ├── cfg.yaml + cfg.json            (full resolved config)
        │   │   ├── epoch_1/, epoch_2/, epoch_3/
        │   │   │   ├── _complete                  (sealed sentinel)
        │   │   │   ├── env_meta.json
        │   │   │   ├── optimizer.pt, scheduler.pt
        │   │   │   └── model.safetensors (+ index/config/tokenizer)
        │   │   ├── metrics.json
        │   │   └── selected_indices_epoch{1,2,3}.json
        │   └── _latest -> runs/<tag>/             (auto-eval reads this)
        ├── data_agent_10/   …
        ├── tads_10/         …
        └── full_100/        …
```

`tads/eval.py` resolves `_latest` automatically, so once `_complete` files
appear under `epoch_3/`, no manual ckpt argument is needed for evaluation.

---

## 6. GPU allocation — two 4-GPU chains, parallel

| Resource         | Chain A                                 | Chain B                              |
|------------------|-----------------------------------------|--------------------------------------|
| GPUs (DDP world) | 0, 1, 2, 3                              | 4, 5, 6, 7                           |
| `--master_port`  | 29501                                   | 29502                                |
| Method order     | `random_10` → `tads_10`                 | `full_100` → `data_agent_10`         |
| Rationale        | Smallest baseline first; then headline. | Heaviest first; then competitor.     |
| Approx wall time | ~1 h + ~3.5 h ≈ **4.5 h**               | ~3.5 h + ~3.5 h ≈ **7 h**            |
| Launcher script  | `scripts/run_evol_qwen_chainA.sh`       | `scripts/run_evol_qwen_chainB.sh`    |
| Tmux session     | `evol_qwen_A`                           | `evol_qwen_B`                        |
| Master log       | `logs/evol_qwen_chainA_master.log`      | `logs/evol_qwen_chainB_master.log`   |

The two scripts are **independent** — chain A and chain B never share a
GPU and use distinct master ports, so a crash in one does not affect the
other. End-to-end wall time is dominated by chain B (~7 h).

---

## 7. Common environment

All chains run inside the project venv:

```bash
source /group-volume/jieuns/llm-instruction-tuning/venv/bin/activate
```

And export thread caps (libgomp `Resource temporarily unavailable` happened
on a previous node when defaults exposed all 247 logical cores per process
to OpenMP/MKL/OpenBLAS):

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
       NUMEXPR_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
```

These are baked into both launchers, so callers don't have to remember
them.

---

## 8. Launch (operator runbook)

```bash
cd /group-volume/jieuns/tads_v2
git pull origin main 2>&1 | tail -3        # fetch this plan + scripts

# Make sure the storage roots exist (writable by jieuns).
mkdir -p /group-volume/jieuns/tads-checkpoints
mkdir -p logs

# Chain A — GPU 0-3
tmux kill-session -t evol_qwen_A 2>/dev/null
tmux new -d -s evol_qwen_A
tmux send-keys -t evol_qwen_A \
  "bash scripts/run_evol_qwen_chainA.sh 2>&1 | tee logs/evol_qwen_chainA_master.log" C-m

# Chain B — GPU 4-7
tmux kill-session -t evol_qwen_B 2>/dev/null
tmux new -d -s evol_qwen_B
tmux send-keys -t evol_qwen_B \
  "bash scripts/run_evol_qwen_chainB.sh 2>&1 | tee logs/evol_qwen_chainB_master.log" C-m

# 90 sec sanity check
sleep 90
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
tail -n 30 logs/evol_qwen_chainA_master.log
tail -n 30 logs/evol_qwen_chainB_master.log
```

A healthy start shows:

- All 8 GPUs holding ≥ 30 GB each (DDP × 4 ranks × 2 chains).
- Two `Loading Alpaca from local file(s): .../wizardlm_evol_instruct_70k/...`
  log lines (one per chain).
- `n=70000 | max_seq_len=512 | style=qwen_chatml` after tokenisation.
- No `libgomp: Thread creation failed` (thread caps applied).

---

## 9. Health monitoring (during run)

```bash
# (a) Per-cell progress
for c in chainA chainB; do
  echo "=== $c ==="
  tail -n 5 /group-volume/jieuns/tads_v2/logs/evol_qwen_${c}_master.log
done

# (b) Per-method run log (replace <method> + <tag>)
ls -lt /group-volume/jieuns/tads_v2/logs/evol_qwen_*.log | head

# (c) sealed epochs so far
find /group-volume/jieuns/tads-checkpoints/evol_7b/qwen25 -name "_complete" \
     -printf "%T@  %p\n" | sort -n | tail

# (d) tmux sessions still alive
tmux ls
```

The training itself logs `SFT | epoch=N | step=… | loss=… | lr=… | mem=…/…GB`
every 50 steps; if `mem` keeps climbing or `loss` spikes / NaNs, kill the
specific tmux session with `tmux kill-session -t evol_qwen_<A|B>` (never
`pkill -9 python` — per CLAUDE.md, that kills the jupyter node too).

---

## 10. Post-training: evaluation

Each cell finishes when sealed `epoch_3` appears with `_complete`. Eval
runs separately, one cell per GPU, against the standard 5-benchmark battery
(`mmlu, gsm8k, humaneval, tydiqa, bbh`):

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> python -m tads.eval \
    --config configs/experiments/evol_7b/qwen25/<method>.yaml \
    --benchmarks mmlu,gsm8k,humaneval,tydiqa,bbh
```

Results land in
`${EVAL_RESULTS_ROOT}/evol_7b/qwen25/<method>/<experiment_label>-<bench>.json`
(`experiment_label = qwen25_<method>` per `tads/eval.py`'s `<parent_stem>_<method>` rule).

`_latest` resolution makes `--ckpt` optional. We can fire the four evals
in parallel on any 4 GPUs that finished training first, **while the rest of
the matrix is still training**, to keep utilisation pinned at 100%.

---

## 11. Failure modes (recovery playbook)

| Symptom in master log                                            | Likely cause                              | Action                                                                                  |
|------------------------------------------------------------------|-------------------------------------------|-----------------------------------------------------------------------------------------|
| `libgomp: Thread creation failed: Resource temporarily unavailable` | Thread cap not applied                 | Restart with the `OMP_NUM_THREADS=4 …` block (already in launcher).                     |
| `CUDA out of memory` during anchor PCA (tads_10 only)            | `max_samples_for_pca` too high           | Edit `tads_10.yaml` to `max_samples_for_pca: 512`, `pca_batch_size: 8` and retry that cell. |
| `NCCL` timeout / hang on init                                    | `--master_port` collision                | Confirm chain A uses 29501, chain B uses 29502 (no other DDP jobs on those ports).      |
| Silent log freeze, GPU memory drops to 0                         | Cluster reclaimed node (too low util)     | Confirm both chains are alive (`nvidia-smi` shows 8 GPUs busy). If yes, restart on the new node from `_latest` of the partially-trained cell. |
| `FileNotFoundError: alpaca_evol_instruct_70k.json`               | dataset path mismatch                     | `ls /group-volume/IT-datasets/wizardlm_evol_instruct_70k/` — confirm filename hasn't changed. |
| `Permission denied: /group-volume/minsoo3.kim/...`               | Leftover colleague path                   | `grep -rn minsoo3 configs scripts` — purge as in PR #2 / #3 history.                    |

Restart-from-sealed-epoch is supported: `tads.train` reads the largest
`_complete` epoch under `_latest` and resumes if the same `--run_tag` is
passed. The launchers always use auto-timestamp tags, so a clean retry
starts a fresh `runs/<new_ts>/` and does **not** clobber prior work.

---

## 12. Paper-side deliverables

| Element                                              | Source artefact                                                |
|------------------------------------------------------|----------------------------------------------------------------|
| Table — "Evol-Instruct row" (Qwen2.5-7B × 4 methods × 5 benches) | `${EVAL_RESULTS_ROOT}/evol_7b/qwen25/*/*-eval_summary.json` |
| Comparison plot — Alpaca vs Evol margin per method   | Combine with `main_7b/qwen25/*` summaries                      |
| §4.x / Discussion — dataset-axis ablation paragraph  | Derived from the two-row matrix (Llama-2 evol + Qwen2.5 evol)  |
| §4 sanity: `tads_10 > random_10` on Evol             | Must hold on every benchmark — otherwise RED flag (§0-2 of AUTO_EVAL_AGENT.md) |

---

## 13. Pointers

| Asset                                  | Path                                                       |
|----------------------------------------|------------------------------------------------------------|
| `random_10` config                     | `configs/experiments/evol_7b/qwen25/random_10.yaml`        |
| `data_agent_10` config                 | `configs/experiments/evol_7b/qwen25/data_agent_10.yaml`    |
| `tads_10` config                       | `configs/experiments/evol_7b/qwen25/tads_10.yaml`          |
| `full_100` config                      | `configs/experiments/evol_7b/qwen25/full_100.yaml`         |
| Launcher — chain A (GPU 0-3)           | `scripts/run_evol_qwen_chainA.sh`                          |
| Launcher — chain B (GPU 4-7)           | `scripts/run_evol_qwen_chainB.sh`                          |
| Counterpart matrix (Alpaca-GPT4)       | `configs/experiments/main_7b/qwen25/*.yaml`                |
| Prior llama2 evol_7b checkpoints       | `/group-volume/jieuns/tads-checkpoints/evol_7b/llama2/*/epoch_3/` (legacy flat layout, from May 14) |
| Data loader (reused)                   | `tads/data/alpaca.py`                                      |
| Prompt template (qwen_chatml)          | `tads/data/sft_prompts.py`                                 |
| Auto-eval agent guide                  | `AUTO_EVAL_AGENT.md`                                       |
| Project memory                         | `CLAUDE.md`                                                |
