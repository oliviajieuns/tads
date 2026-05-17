"""Theorem 1 verifier smoke tests — mock model, no GPU, no real dataset.

Covers:
  * TrajectoryAnchor: track_sigma path computes ‖ΔΣ‖_F finite from step 2
  * TrajectoryAnchor: last_measurement is populated with d_step,
    sign_inner_prev, gamma per layer
  * TheoremVerifier: refresh cadence + JSONL output + .npy dumps
  * analyze_theorem.py: end-to-end on synthetic JSONL produces PASS verdicts
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from tads.core.thm_verification import (
    TheoremVerificationConfig,
    TheoremVerifier,
)
from tads.core.trajectory_anchor import TrajectoryAnchor


# --------------------------------------------------------------- mock model
class _MockOutput:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class MockModel(torch.nn.Module):
    """Two-layer toy that has linear-in-input hidden states.

    The hidden states are a function of a small learnable parameter so
    repeatedly nudging that parameter mimics training-induced drift in
    Δh, and therefore in the per-layer covariance Σ_l.
    """
    def __init__(self, vocab: int = 32, hidden: int = 8, n_layers: int = 3):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, hidden)
        # A bias per layer that we'll mutate to simulate training drift.
        self.layer_bias = torch.nn.Parameter(torch.zeros(n_layers, hidden))
        self.n_layers = n_layers

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, **kw):
        h0 = self.embed(input_ids)  # (B, T, H)
        hs = [h0]
        cur = h0
        for li in range(self.n_layers):
            cur = cur + self.layer_bias[li].view(1, 1, -1)
            hs.append(cur)
        return _MockOutput(tuple(hs))


class MockDataset(Dataset):
    def __init__(self, n: int = 32, seq_len: int = 6, vocab: int = 32, seed: int = 0):
        g = torch.Generator(); g.manual_seed(seed)
        self.ids = torch.randint(0, vocab, (n, seq_len), generator=g)
        self.mask = torch.ones_like(self.ids)

    def __len__(self) -> int:
        return self.ids.shape[0]

    def __getitem__(self, i: int):
        return {"input_ids": self.ids[i], "attention_mask": self.mask[i]}


# --------------------------------------------------------------- anchor tests
def test_anchor_tracks_delta_sigma_and_d_step():
    model = MockModel()
    dataset = MockDataset()
    anchor = TrajectoryAnchor(
        layer_indices="all",
        max_samples_for_pca=16,
        pca_batch_size=4,
        device="cpu",
        track_sigma=True,
    )
    # t=0
    anchor.update(model=model, dataset=dataset, seed=0, epoch=0,
                  global_step=0, lr=1e-3, probe_seed_override=99)
    m0 = anchor.last_measurement
    # First call has no prior — ΔΣ and d should be NaN.
    for li, row in m0["per_layer"].items():
        assert not np.isfinite(row["delta_sigma_fro"]), \
            f"layer {li}: ΔΣ should be NaN at t=0, got {row['delta_sigma_fro']}"
        assert not np.isfinite(row["d_step"]), \
            f"layer {li}: d should be NaN at t=0"
        assert row["gamma"] >= 0, "λ1 should be ≥ λ2"

    # Nudge the model to simulate one optimizer step.
    with torch.no_grad():
        model.layer_bias.add_(torch.randn_like(model.layer_bias) * 0.1)

    anchor.update(model=model, dataset=dataset, seed=0, epoch=0,
                  global_step=10, lr=1e-3, probe_seed_override=99)
    m1 = anchor.last_measurement
    for li, row in m1["per_layer"].items():
        assert np.isfinite(row["delta_sigma_fro"]), \
            f"layer {li}: ΔΣ must be finite at t=1"
        assert np.isfinite(row["d_step"]), \
            f"layer {li}: d must be finite at t=1"
        assert row["d_step"] >= 0
        # Sign inner: anchor.update applies sign calibration, then we
        # compare to the previous (calibrated) vector. For a small param
        # nudge it should not flip; assert > -1 + slack.
        assert row["sign_inner_prev"] > -0.999


def test_anchor_restores_train_mode():
    model = MockModel()
    model.train()
    dataset = MockDataset()
    anchor = TrajectoryAnchor(
        layer_indices="all",
        max_samples_for_pca=8, pca_batch_size=4, device="cpu",
    )
    anchor.update(model=model, dataset=dataset, seed=0, epoch=0)
    assert model.training, "anchor.update() must restore train mode"

    model.eval()
    anchor.update(model=model, dataset=dataset, seed=0, epoch=0)
    assert not model.training, "anchor.update() must preserve eval mode if it was eval"


# --------------------------------------------------------------- verifier
def test_verifier_refresh_cadence_and_outputs(tmp_path: Path):
    model = MockModel()
    dataset = MockDataset()
    anchor = TrajectoryAnchor(
        layer_indices="all", max_samples_for_pca=8,
        pca_batch_size=4, device="cpu",
    )
    cfg = TheoremVerificationConfig(
        enabled=True,
        refresh_every_optstep=2,    # fire on cumulative steps 2, 4, 6, ...
        save_anchors=True,
        track_delta_sigma=True,
        probe_size=8,
        record_baseline=True,
    )
    out_dir = tmp_path / "ver"
    verifier = TheoremVerifier(
        cfg=cfg, anchor=anchor, dataset=dataset,
        output_dir=out_dir, seed=0,
    )
    verifier.open()
    try:
        # baseline (step 0)
        verifier.baseline(model, lr=1e-3, epoch=1)
        # Simulate 6 optimizer steps with a tiny weight nudge between each.
        for i in range(6):
            with torch.no_grad():
                model.layer_bias.add_(torch.randn_like(model.layer_bias) * 0.05)
            verifier.step(model, lr=1e-3 * (1.0 - i / 20), epoch=1)
    finally:
        verifier.close()

    # JSONL: baseline (step=0) + refreshes at cumulative steps 2,4,6 → 4 refreshes
    rows = [json.loads(l) for l in (out_dir / "metrics.jsonl").read_text().splitlines() if l.strip()]
    steps = sorted({int(r["global_step"]) for r in rows})
    assert steps == [0, 2, 4, 6], f"unexpected steps {steps}"

    # Anchors dumped per refresh, one .npy per step.
    npys = sorted((out_dir / "anchors").glob("step_*.npy"))
    assert len(npys) == 4

    arr = np.load(npys[1])
    # (n_layers, H) = (3, 8) for MockModel
    assert arr.shape == (3, 8)

    # layer_indices.json present
    li = json.loads((out_dir / "anchors" / "layer_indices.json").read_text())
    assert li == [0, 1, 2]


def test_verifier_inactive_is_noop(tmp_path: Path):
    """When cfg.enabled=False, verifier.step() must not write anything."""
    cfg = TheoremVerificationConfig(enabled=False)
    anchor = TrajectoryAnchor(layer_indices="all", device="cpu",
                              max_samples_for_pca=4, pca_batch_size=2)
    verifier = TheoremVerifier(
        cfg=cfg, anchor=anchor, dataset=MockDataset(),
        output_dir=tmp_path / "ver", seed=0,
    )
    verifier.open()
    verifier.step(MockModel(), lr=1e-3, epoch=0)
    verifier.close()
    assert not (tmp_path / "ver" / "metrics.jsonl").exists()


# --------------------------------------------------------------- analyzer
def _load_analyzer():
    """Import scripts/analyze_theorem.py as a module (not on sys.path)."""
    here = Path(__file__).resolve().parent.parent
    path = here / "scripts" / "analyze_theorem.py"
    spec = importlib.util.spec_from_file_location("analyze_theorem", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _write_synthetic_run(run_dir: Path, n_steps: int = 25, n_layers: int = 3):
    """Build a synthetic metrics.jsonl that PERFECTLY satisfies Theorem 1:
      d_l^(t) = 0.4 · bound,  γ ≡ 0.5,  ‖ΔΣ‖_F = C·η,  sign-inner ≡ 1.
    Used to confirm the analyzer reports PASS on a clean signal."""
    ver = run_dir / "thm_verification"
    ver.mkdir(parents=True)
    (ver / "anchors").mkdir()
    lr0 = 2e-4
    C_sigma_true = 0.3
    gamma = 0.5
    bound_factor = 2 * C_sigma_true / gamma  # = 1.2
    rows = []
    for t in range(n_steps):
        lr = lr0 * (1 - t / (2 * n_steps))  # cosine-ish decay
        for li in range(n_layers):
            d = 0.4 * bound_factor * lr
            ds = C_sigma_true * lr
            rows.append({
                "global_step": t,
                "epoch": 1,
                "lr": lr,
                "layer_idx": li,
                "lambda1": 1.0,
                "lambda2": 1.0 - gamma,
                "gamma": gamma,
                "delta_sigma_fro": ds if t > 0 else float("nan"),
                "sign_inner_prev": 1.0 if t > 0 else float("nan"),
                "d_step": d if t > 0 else float("nan"),
            })
    with open(ver / "metrics.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_analyzer_end_to_end_pass(tmp_path: Path):
    mod = _load_analyzer()
    run_dir = tmp_path / "run"
    _write_synthetic_run(run_dir)
    summary = mod.run(run_dir)
    v = summary["verdicts"]
    assert v["E1"] == "PASS", v
    assert v["E2"] == "PASS", v
    assert v["E3"] == "PASS", v
    assert v["E3b"] == "PASS", v
    assert v["E4"] == "PASS", v
    # Spot-check C_Σ recovery within 5 %
    fitted = summary["E2_C_sigma"]["C_sigma"]
    assert abs(fitted - 0.3) / 0.3 < 0.05, fitted
    # Tightness ratio ≈ 0.4
    assert 0.35 < summary["E3_per_step_bound"]["tightness"]["mean"] < 0.45
    # Figures exist
    fig_dir = run_dir / "thm_verification" / "analysis" / "figures"
    for stem in ("F1_eigengap_heatmap", "F2_dSigma_vs_lr",
                 "F3_d_vs_bound", "F4_cumulative_S"):
        assert (fig_dir / f"{stem}.png").exists(), stem
        assert (fig_dir / f"{stem}.pdf").exists(), stem
