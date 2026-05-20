# Porting the Theorem 1 verifier from oliviajieuns/tads to 33modeling/tads

This bundle ports the per-step anchor verifier (App. F / Fig. 2
instrumentation) from the oliviajieuns fork into the 33modeling main
branch. Four code files change; three new YAML configs add the verifier
to the 7B training pipeline.

**Target repo**: `/group-volume/jieuns/tads_33m/`
**Source bundle root** (after `unzip thm_verify_7b_port.zip`): `./thm_verify_7b_port/`

## Step 1 — drop in the new module (no merge needed)

```
cp thm_verify_7b_port/core/thm_verification.py \
   /group-volume/jieuns/tads_33m/tads/core/thm_verification.py
```

Self-contained. Imports only from `trajectory_anchor.py` and `utils.py`
(both already in 33m).

## Step 2 — replace trajectory_anchor.py (or merge if 33m has unique additions)

The olivia version adds ~70 lines for verifier support:
  - constructor arg `track_sigma: bool = False`
  - field `last_measurement: Dict[str, Any]`
  - field `v_by_layer: Dict[int, torch.Tensor]`
  - `update()` signature accepts `global_step, lr, probe_seed_override` (all Optional)
  - inside `update()`, Σ tracking gated by `self.track_sigma`
  - after `update()`, `self.last_measurement` populated with per-layer metrics

Safest path — diff first, then decide:

```
cd /group-volume/jieuns/tads_33m
diff tads/core/trajectory_anchor.py \
     ../thm_verify_7b_port/core/trajectory_anchor.py | less
```

If 33m has no new logic beyond what olivia has, just replace:

```
cp thm_verify_7b_port/core/trajectory_anchor.py \
   /group-volume/jieuns/tads_33m/tads/core/trajectory_anchor.py
```

If 33m has unique additions (e.g. padding-invariant aggregation, new
layer-selection logic), manually merge: keep the 33m core and add the
six verifier-extension points listed above.

## Step 3 — patch sft.py: add verifier hook in sft_one_epoch

Two edits.

(a) Function signature (around line ~101 in olivia, similar in 33m):

```python
def sft_one_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scheduler,
    grad_accum: int,
    grad_clip: float,
    device,
    epoch: int,
    logger: Optional[logging.Logger] = None,
    log_every: int = 50,
    verifier: Optional[object] = None,     # <-- ADD
) -> float:
```

(b) After `optimizer.step()` and `optimizer.zero_grad(...)`, add the
verifier hook block (around line ~232 in olivia):

```python
            # Theorem 1 verifier hook — fires every N optimizer steps on
            # rank 0 only (the verifier itself is the gatekeeper, see
            # TheoremVerifier.active). Outside the verification path this
            # is a no-op.
            if verifier is not None:
                try:
                    verifier.step(
                        model=model,
                        lr=scheduler.get_last_lr()[0],
                        epoch=epoch,
                    )
                except Exception as exc:
                    # Verification must not crash the training run.
                    logger.warning("Theorem verifier refresh failed: %s", exc)
```

If 33m's sft.py is structurally similar to olivia's, just replace:

```
cp thm_verify_7b_port/pipelines/sft.py \
   /group-volume/jieuns/tads_33m/tads/pipelines/sft.py
```

Then `diff` it back to recover any 33m-only changes.

## Step 4 — patch train.py: init / open / pass / close

Three edits.

(a) Near the top imports (olivia line 92):

```python
from tads.core.thm_verification import (
    TheoremVerificationConfig, TheoremVerifier,
)
```

(b) After dataset is loaded and `anchor` is constructed, BEFORE the
training loop (olivia lines 639-659):

```python
    # ---------- Theorem 1 verifier (App. F) ----------
    verifier_cfg = TheoremVerificationConfig.from_cfg(cfg.get("verification"))
    verifier: Optional[TheoremVerifier] = None
    # NOTE: olivia gates on `method == "tads"` only. For the matched
    # no-anchor (lambda=0) control we want the diagnostic too, so drop
    # that gate — the verifier itself no-ops when anchor is None.
    if verifier_cfg.enabled and anchor is not None:
        verifier = TheoremVerifier(
            cfg=verifier_cfg,
            anchor=anchor,
            dataset=dataset,
            output_dir=run_dir / verifier_cfg.output_subdir,
            seed=seed,
        )
        if is_main_process():
            verifier.open()
            logger.info(
                "Theorem 1 verifier ENABLED | output=%s",
                run_dir / verifier_cfg.output_subdir,
            )
```

(c) Pass to `sft_one_epoch` (olivia line 713 — single arg addition):

```python
        avg_loss = sft_one_epoch(
            ...                       # existing args
            verifier=verifier,        # <-- ADD
        )
```

(d) After the training loop closes (olivia lines 880-881):

```python
    if verifier is not None and is_main_process():
        verifier.close()
```

If 33m's train.py is structurally similar, full replacement is safe.

## Step 5 — drop in the 3 new YAML configs

```
cd /group-volume/jieuns/tads_33m
mkdir -p configs/experiments/thm_verify_7b
cp ../thm_verify_7b_port/configs/B_probe2000_cosine.yaml      configs/experiments/thm_verify_7b/
cp ../thm_verify_7b_port/configs/D_probe2000_constant.yaml    configs/experiments/thm_verify_7b/
cp ../thm_verify_7b_port/configs/no_anchor_cosine.yaml        configs/experiments/thm_verify_7b/
```

## Step 6 — sanity test (30 sec dry-run)

Before kicking off a 12-hour training, verify the verifier imports and
the YAML loads:

```bash
cd /group-volume/jieuns/tads_33m
source /group-volume/jieuns/llm-instruction-tuning/venv/bin/activate
python -c "
from tads.core.thm_verification import TheoremVerificationConfig, TheoremVerifier
from tads.core.trajectory_anchor import TrajectoryAnchor
print('imports OK')
print('TrajectoryAnchor supports track_sigma:',
      'track_sigma' in TrajectoryAnchor.__init__.__code__.co_varnames)
print('TrajectoryAnchor.update accepts global_step:',
      'global_step' in TrajectoryAnchor.update.__code__.co_varnames)
"

# YAML round-trip via OmegaConf
python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('configs/experiments/thm_verify_7b/B_probe2000_cosine.yaml')
print('verification block:', dict(cfg.get('verification', {})))
print('anchor block      :', dict(cfg.get('anchor', {})))
"
```

If both prints succeed, ports are wired correctly. If `track_sigma`
prints `False` (i.e. the constructor doesn't accept it), step 2 didn't
take — re-check the merge.

## Step 7 — launch (after sanity passes)

```bash
# 1108843 (80GB full) — cell B
nohup python -m tads.train \
  --config configs/experiments/thm_verify_7b/B_probe2000_cosine.yaml \
  > /tmp/thm_verify_7b_cellB.log 2>&1 &
sleep 120 && tail -50 /tmp/thm_verify_7b_cellB.log

# 1108842 (80GB full) — cell D
nohup python -m tads.train \
  --config configs/experiments/thm_verify_7b/D_probe2000_constant.yaml \
  > /tmp/thm_verify_7b_cellD.log 2>&1 &

# MIG node — no-anchor control
nohup python -m tads.train \
  --config configs/experiments/thm_verify_7b/no_anchor_cosine.yaml \
  > /tmp/thm_verify_7b_noanc.log 2>&1 &
```

Output anchors land at:
```
<output_root>/thm_verify_7b/qwen25/<cell>/runs/<tag>/thm_verification/
  metrics.jsonl
  anchors/
    step_00000025.npy ...
```

## Step 8 — post-run figure regeneration

Use the existing `scripts/extract_thm_main_tight.py` from olivia (or
its bundled copy) on the cell-B output to regenerate the cumulative
tail-bound figure with 7B numbers.
