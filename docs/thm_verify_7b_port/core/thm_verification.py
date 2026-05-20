"""Theorem 1 verification harness.

Hooks into the SFT loop to refresh the TrajectoryAnchor every
``refresh_every_optstep`` optimizer steps and dumps per-step / per-layer
measurements to JSONL. The recorded fields cover:

  - assumption A1 (Σ-drift bounded by lr):  ‖Σ^(t+1) − Σ^(t)‖_F  vs.  η_t
  - assumption A2 (eigengap positive):       λ1 − λ2
  - assumption A3 (sign calibration):        ⟨v^(t), v^(t+1)⟩
  - conclusion C1 (per-step bound):          d^(t) = ‖v^(t+1) − v^(t)‖

The anchor vectors v_l^(t) are stored separately as .npy files so the
offline analyser can compute cumulative bound (conclusion C2) without
having to re-run training.

The verifier is rank-0 only — anchor extraction is single-rank in the
rest of the codebase (see selection.py), and step-level refresh inherits
that contract. DDP workers see the verifier as a no-op.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Subset

from .trajectory_anchor import TrajectoryAnchor
from .utils import is_main_process

logger = logging.getLogger(__name__)


@dataclass
class TheoremVerificationConfig:
    """User-facing knobs (mirrors the ``verification:`` block in cfg YAML)."""
    enabled: bool = False
    refresh_every_optstep: int = 50
    save_anchors: bool = True
    track_delta_sigma: bool = True
    probe_size: Optional[int] = None         # falls back to anchor.max_samples_for_pca
    probe_seed: int = 4242                   # FIXED across refreshes (see anchor.update)
    output_subdir: str = "thm_verification"  # under run_dir
    # Whether to also do a t=0 baseline refresh BEFORE the first optimizer
    # step. Useful: without it the first measured row has d^(t)=NaN and
    # ΔΣ=NaN (no prior state) and the verifier gets one fewer datapoint.
    record_baseline: bool = True

    @classmethod
    def from_cfg(cls, cfg_block: Optional[Dict[str, Any]]) -> "TheoremVerificationConfig":
        if not cfg_block:
            return cls()
        kw: Dict[str, Any] = {}
        for k in ("enabled", "refresh_every_optstep", "save_anchors",
                  "track_delta_sigma", "probe_size", "probe_seed",
                  "output_subdir", "record_baseline"):
            if k in cfg_block:
                kw[k] = cfg_block[k]
        return cls(**kw)


@dataclass
class TheoremVerifier:
    """Owns the JSONL writer + the fixed probe Subset.

    Construct ONCE per training run (in train.py). Pass into sft_one_epoch.
    The verifier silently no-ops on non-rank-0 and when ``cfg.enabled`` is
    False, so callers don't need to guard.
    """
    cfg: TheoremVerificationConfig
    anchor: Optional[TrajectoryAnchor]
    dataset: Any
    output_dir: Path
    seed: int
    metrics_path: Path = field(init=False)
    anchors_dir: Path = field(init=False)
    _probe_subset: Optional[Subset] = field(default=None, init=False)
    _file: Any = field(default=None, init=False)
    _n_refreshes: int = field(default=0, init=False)
    # Cumulative optimizer-step counter across ALL epochs of this run.
    # Owned by the verifier so callers don't need to bookkeep an offset.
    cumulative_optstep: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.anchors_dir = self.output_dir / "anchors"

    # --------------------------------------------------------- lifecycle
    @property
    def active(self) -> bool:
        return bool(self.cfg.enabled and self.anchor is not None and is_main_process())

    def open(self) -> None:
        if not self.active:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.save_anchors:
            self.anchors_dir.mkdir(parents=True, exist_ok=True)
        # Append-mode: a mid-epoch crash + resume keeps the prior rows.
        self._file = open(self.metrics_path, "a", buffering=1)
        # Wire Σ tracking into the anchor instance. We set it here rather
        # than in __init__ so a single TrajectoryAnchor can be (re-)used
        # by both the per-epoch selection path AND the per-step verifier
        # path without forcing track_sigma=True onto runs that don't need
        # it (the host-RAM cost is non-trivial for 7B).
        self.anchor.track_sigma = bool(self.cfg.track_delta_sigma)
        # Fixed probe: subsample the dataset once with a stable seed so
        # every refresh sees the SAME inputs — Σ drift then reflects only
        # the model parameters.
        n = len(self.dataset)
        probe_n = int(self.cfg.probe_size
                      or self.anchor.max_samples_for_pca)
        probe_n = min(probe_n, n)
        g = torch.Generator(); g.manual_seed(int(self.cfg.probe_seed))
        perm = torch.randperm(n, generator=g).tolist()[:probe_n]
        self._probe_subset = Subset(self.dataset, perm)
        logger.info(
            "[thm_verifier] OPEN | metrics=%s | probe_size=%d | "
            "refresh_every=%d | track_sigma=%s",
            self.metrics_path, probe_n, self.cfg.refresh_every_optstep,
            self.cfg.track_delta_sigma,
        )

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                pass
            self._file = None

    # --------------------------------------------------------- refresh
    def should_refresh(self) -> bool:
        if not self.active:
            return False
        if self.cumulative_optstep <= 0:
            return False
        return (self.cumulative_optstep % self.cfg.refresh_every_optstep) == 0

    def baseline(self, model, *, lr: float, epoch: int) -> None:
        """Optional t=0 measurement BEFORE the first optimizer step.

        Without this, the first row in metrics.jsonl has d=NaN and ΔΣ=NaN
        (no prior anchor / Σ to diff against), so the verifier effectively
        loses one datapoint. record_baseline=True (default) trades one
        extra PCA pass for that datapoint.
        """
        if not self.active:
            return
        if not self.cfg.record_baseline:
            return
        self._refresh(model, global_optstep=0, lr=lr, epoch=epoch)

    def step(
        self,
        model,
        *,
        lr: float,
        epoch: int,
    ) -> None:
        """Call once per optimizer.step(). Bumps the cumulative counter and
        triggers a refresh on the configured cadence. No-op when inactive."""
        if not self.active:
            return
        self.cumulative_optstep += 1
        if self.should_refresh():
            self._refresh(
                model,
                global_optstep=self.cumulative_optstep,
                lr=lr, epoch=epoch,
            )

    def _refresh(self, model, *, global_optstep: int, lr: float, epoch: int) -> None:
        assert self.anchor is not None and self._probe_subset is not None
        self.anchor.update(
            model=model,
            dataset=self._probe_subset,
            seed=self.seed,
            epoch=epoch,
            global_step=global_optstep,
            lr=lr,
            probe_seed_override=self.cfg.probe_seed,
        )
        m = self.anchor.last_measurement
        rows: list = []
        for li, layer_metrics in m["per_layer"].items():
            rows.append({
                "global_step": global_optstep,
                "epoch": epoch,
                "lr": lr,
                "layer_idx": int(li),
                **{k: float(v) for k, v in layer_metrics.items()},
            })
        for r in rows:
            self._file.write(json.dumps(r) + "\n")

        if self.cfg.save_anchors:
            # Flat (L, H) stack ordered by layer_idx for deterministic
            # downstream concatenation.
            sorted_li = sorted(m["v_by_layer"].keys())
            arr = np.stack(
                [m["v_by_layer"][li].cpu().numpy() for li in sorted_li],
                axis=0,
            )
            np.save(
                self.anchors_dir / f"step_{global_optstep:08d}.npy",
                arr.astype(np.float32),
            )
            # Save the layer-index ordering once so the analyzer can map
            # rows back to layers without guessing.
            idx_path = self.anchors_dir / "layer_indices.json"
            if not idx_path.exists():
                idx_path.write_text(json.dumps(sorted_li))

        self._n_refreshes += 1
        logger.info(
            "[thm_verifier] refresh #%d | step=%d | lr=%.3e | rows=%d",
            self._n_refreshes, global_optstep, lr, len(rows),
        )

    # context-manager sugar so callers don't have to remember open/close.
    def __enter__(self) -> "TheoremVerifier":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
