"""Trajectory Anchor — multi-layer capability direction (NAIT Eq 2 + Eq 5).

For each layer l and sample x, the contextualization vector is
    Δh_l(x; θ_t) := h_l^last(x; θ_t) - h_l^first(x; θ_t).
For each layer we PCA the set {Δh_l(x; θ_t)} over a probe subset and take
the top-1 eigenvector as that layer's capability direction v_l (sign
calibrated so ⟨v_l, E[Δh_l]⟩ > 0).

At scoring time NAIT (Eq 5) aggregates across layers:
    s_y = Σ_{l ∈ layer_indices} ⟨Δh_l(y), v_l⟩

Layer selection (``layer_indices`` arg):
    "all"             → every decoder layer (NAIT paper, recommended)
    "middle_to_last"  → layers L//2 .. L-1 (memory-friendlier ablation)
    list[int]         → explicit indices
    None              → falls back to legacy single-layer mode using ``layer_idx``
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)


LayerSpec = Union[str, List[int], None]

# Maximum number of past epochs of (v, lambda, stability) history we keep
# in memory. NAIT-faithful runs do 3 epochs so this is effectively unlimited;
# the cap matters only for long ablation sweeps that call .update() many
# times against the same anchor instance.
_MAX_HISTORY = 50


def _resolve_layer_indices(spec: LayerSpec, num_decoder_layers: int) -> List[int]:
    """Translate a layer-spec into a concrete list of decoder-layer indices.

    ``num_decoder_layers`` is the number of transformer blocks (L). Returned
    indices are 0-based among the L decoder layers (NOT among hidden_states
    which has L+1 entries because position 0 is the embedding).
    """
    if spec is None:
        raise ValueError("layer_indices spec is None — caller should fall back to legacy mode")
    if isinstance(spec, str):
        if spec == "all":
            return list(range(num_decoder_layers))
        if spec == "middle_to_last":
            return list(range(num_decoder_layers // 2, num_decoder_layers))
        raise ValueError(f"Unknown layer_indices string: {spec!r}")
    if isinstance(spec, (list, tuple)):
        out = [int(x) for x in spec]
        for i in out:
            if not (0 <= i < num_decoder_layers):
                raise ValueError(
                    f"layer index {i} out of range [0, {num_decoder_layers}); "
                    f"model has {num_decoder_layers} decoder layers."
                )
        return out
    raise TypeError(f"layer_indices must be str | list | None, got {type(spec).__name__}")


class TrajectoryAnchor:
    """Multi-layer capability anchor.

    ``layer_indices`` can be passed at construction time (resolved against
    the model's layer count at the first :meth:`update` call), or left as
    None to use the legacy single-layer behaviour driven by ``layer_idx``.
    """

    def __init__(
        self,
        layer_idx: int = -1,
        layer_indices: LayerSpec = None,
        max_samples_for_pca: int = 2000,
        pca_batch_size: int = 4,
        device: str = "cuda",
        track_sigma: bool = False,
    ):
        self.layer_idx = layer_idx
        self.layer_indices_spec: LayerSpec = layer_indices
        self.max_samples_for_pca = max_samples_for_pca
        self.pca_batch_size = pca_batch_size
        self.device = device

        # Theorem 1 verification: when True, keep the previous covariance Σ_l
        # per layer in host RAM so update() can report ‖Σ^(t+1)−Σ^(t)‖_F
        # (assumption A1 measurement). Cost: L × H² fp32 floats on CPU.
        # For 0.5B (L≈24, H=896) ≈80 MB; for 7B (L=32, H=4096) ≈2 GB —
        # leave OFF unless explicitly running the App. F verification.
        self.track_sigma = bool(track_sigma)
        self.prev_sigma_by_layer: Dict[int, torch.Tensor] = {}

        # Resolved indices — filled in lazily on first update() so we can
        # use the model's actual layer count. Empty means "not yet resolved
        # OR legacy single-layer mode".
        self.layer_indices: List[int] = []
        # Direction per layer (decoder-layer index → unit vector ∈ R^H).
        self.v_by_layer: Dict[int, torch.Tensor] = {}
        self.lambda1_by_layer: Dict[int, float] = {}
        self.lambda2_by_layer: Dict[int, float] = {}
        self.gap_by_layer: Dict[int, float] = {}

        # Last refresh's full per-layer measurement payload (Theorem 1 verifier
        # reads this after every update() call). Populated by update().
        self.last_measurement: Dict[str, Any] = {}

        # Legacy single-layer state (kept for backward compat).
        self.v: Optional[torch.Tensor] = None  # alias of v_by_layer for single-layer mode
        self.lambda_1: float = 0.0
        self.lambda_2: float = 0.0
        self.gap: float = 0.0

        # History (Theorem 1 verification). Track mean over layers in multi mode.
        self.v_history: List[torch.Tensor] = []
        self.lambda1_history: List[float] = []
        self.lambda2_history: List[float] = []
        self.gap_history: List[float] = []
        self.stability_history: List[float] = []

    @property
    def is_fitted(self) -> bool:
        return bool(self.v_by_layer)

    @property
    def is_multi_layer(self) -> bool:
        return self.layer_indices_spec is not None

    # ------------------------------------------------------------------ PCA
    @staticmethod
    def _pca_top1(delta: torch.Tensor) -> Dict[str, Any]:
        """Run top-1 PCA on a centred (N, H) delta matrix.

        Returns dict with ``v`` (unit eigenvector), ``lambda_1``, ``lambda_2``,
        and the un-centred mean ``mu`` used for sign calibration upstream.
        """
        N, H = delta.shape
        mu = delta.mean(dim=0, keepdim=True)
        centred = delta - mu
        if N < H:
            gram = centred @ centred.T / N
            eigvals, eigvecs = torch.linalg.eigh(gram)
            lambda_1 = float(eigvals[-1].item())
            lambda_2 = float(eigvals[-2].item()) if N >= 2 else 0.0
            top_u = eigvecs[:, -1]
            v = centred.T @ top_u
            v = v / (v.norm() + 1e-8)
        else:
            cov = centred.T @ centred / N
            eigvals, eigvecs = torch.linalg.eigh(cov)
            lambda_1 = float(eigvals[-1].item())
            lambda_2 = float(eigvals[-2].item())
            v = eigvecs[:, -1]
        return {
            "v": v,
            "lambda_1": lambda_1,
            "lambda_2": lambda_2,
            "mu": mu.squeeze(0),
        }

    # ------------------------------------------------------------------ update
    @torch.no_grad()
    def update(
        self,
        model,
        dataset,
        seed: int = 42,
        epoch: int = 0,
        global_step: Optional[int] = None,
        lr: Optional[float] = None,
        probe_seed_override: Optional[int] = None,
    ) -> Dict[str, float]:
        """Re-extract the anchor at the start of epoch ``t``.

        Collects Δh per layer over a probe subset and PCAs each independently.

        Probe seed is offset (+1) from the (seed + epoch*100) used by
        ``_random_indices``. Without the offset both RNGs would draw the
        same permutation, so for ratio≤probe_size/N the random-method
        selection and the anchor probe would overlap perfectly — biasing
        the alignment direction toward the very samples that will then
        be SFT'd, an unintended coupling.
        """
        # Theorem 1 verification uses a FIXED probe (same indices for all
        # refresh points) so that Σ^(t+1) − Σ^(t) reflects ONLY the model
        # drift, not probe resampling. Callers in that mode pass a stable
        # ``probe_seed_override``; default behavior keeps the legacy per-epoch
        # reshuffle so non-verification training is unaffected.
        g = torch.Generator()
        if probe_seed_override is not None:
            g.manual_seed(int(probe_seed_override))
        else:
            g.manual_seed(seed + epoch * 100 + 1)
        was_training = model.training
        n_total = len(dataset)
        n_use = min(self.max_samples_for_pca, n_total)
        perm = torch.randperm(n_total, generator=g).tolist()
        indices = perm[:n_use]

        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=self.pca_batch_size,
            shuffle=False,
            num_workers=0,
        )

        model.eval()
        # Accumulator: layer_idx -> List[Tensor (B, H)].
        per_layer_deltas: Dict[int, List[torch.Tensor]] = {}
        resolved_indices: Optional[List[int]] = None

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states  # tuple, length L+1

            # Decoder layers occupy hidden_states[1:]. Their count is L.
            num_decoder_layers = len(hidden_states) - 1

            # Resolve layer_indices once we know L.
            if resolved_indices is None:
                if self.is_multi_layer:
                    resolved_indices = _resolve_layer_indices(
                        self.layer_indices_spec, num_decoder_layers,
                    )
                else:
                    # Legacy single-layer mode: translate layer_idx (negative
                    # supported, indexing into the full hidden_states tuple
                    # for backward-compat) to a decoder-layer index.
                    li = self.layer_idx
                    if li < 0:
                        li = len(hidden_states) + li
                    # Convert hidden_states-index → decoder-layer-index by
                    # subtracting 1 (embedding offset). When the legacy
                    # spec asks for the embedding itself (li == 0), clamp.
                    decoder_li = max(0, li - 1)
                    resolved_indices = [decoder_li]
                self.layer_indices = resolved_indices

            lengths = (attention_mask.sum(dim=1).clamp_min(1) - 1).to(input_ids.device)
            bidx = torch.arange(input_ids.size(0), device=input_ids.device)

            for li in resolved_indices:
                # hidden_states[li + 1] is the li-th decoder layer
                h = hidden_states[li + 1]                       # (B, T, H)
                first_h = h[:, 0, :]
                last_h = h[bidx, lengths]
                delta = (last_h - first_h).detach().float().cpu()
                per_layer_deltas.setdefault(li, []).append(delta)

            del hidden_states, outputs

        # Per-layer PCA + sign calibration.
        new_v_by_layer: Dict[int, torch.Tensor] = {}
        new_l1: Dict[int, float] = {}
        new_l2: Dict[int, float] = {}
        # Theorem 1 verification per-layer side-channel metrics.
        delta_sigma_fro: Dict[int, float] = {}
        sign_inner_prev: Dict[int, float] = {}
        d_step: Dict[int, float] = {}
        n_used = 0
        for li in resolved_indices or []:
            delta_l = torch.cat(per_layer_deltas[li], dim=0)  # (N, H)
            n_used = delta_l.shape[0]
            pca = self._pca_top1(delta_l)
            v_l = pca["v"]
            # Sign calibration: ⟨v_l, μ_l⟩ > 0.
            if torch.dot(v_l, pca["mu"]) < 0:
                v_l = -v_l
            new_v_by_layer[li] = v_l
            new_l1[li] = pca["lambda_1"]
            new_l2[li] = pca["lambda_2"]

            # Anchor drift d_l^(t) = ‖v_l^(t+1) − v_l^(t)‖ — verifier's
            # left-hand side of the Theorem 1 bound. Computed here so the
            # measurement is co-located with the v that produced it.
            if li in self.v_by_layer:
                d_step[li] = float(torch.norm(v_l - self.v_by_layer[li]).item())
                sign_inner_prev[li] = float(torch.dot(v_l, self.v_by_layer[li]).item())
            else:
                d_step[li] = float("nan")
                sign_inner_prev[li] = float("nan")

            # ‖Σ^(t+1) − Σ^(t)‖_F — verifier's A1 measurement. We compute
            # Σ = X^T X / N from the centred delta only when track_sigma is
            # on (the host-RAM hit for storing prev Σ scales as L × H²; see
            # __init__ docstring). When off, expose NaN so downstream code
            # can still write a row.
            if self.track_sigma:
                centred = delta_l - delta_l.mean(dim=0, keepdim=True)
                sigma_new = (centred.T @ centred) / max(1, centred.shape[0])
                prev = self.prev_sigma_by_layer.get(li)
                if prev is not None:
                    delta_sigma_fro[li] = float(torch.norm(sigma_new - prev).item())
                else:
                    delta_sigma_fro[li] = float("nan")
                self.prev_sigma_by_layer[li] = sigma_new
            else:
                delta_sigma_fro[li] = float("nan")

        # Stability: mean L2 distance between old and new v per layer.
        if self.is_fitted and set(new_v_by_layer.keys()) == set(self.v_by_layer.keys()):
            diffs = [
                float(torch.norm(new_v_by_layer[k] - self.v_by_layer[k]).item())
                for k in new_v_by_layer
            ]
            stability = sum(diffs) / len(diffs) if diffs else float("nan")
        else:
            stability = float("nan")

        # Commit new state.
        self.v_by_layer = new_v_by_layer
        self.lambda1_by_layer = new_l1
        self.lambda2_by_layer = new_l2
        self.gap_by_layer = {k: new_l1[k] - new_l2[k] for k in new_l1}

        # Aggregate scalars for history / logging.
        self.lambda_1 = sum(new_l1.values()) / len(new_l1) if new_l1 else 0.0
        self.lambda_2 = sum(new_l2.values()) / len(new_l2) if new_l2 else 0.0
        self.gap = self.lambda_1 - self.lambda_2

        # Single-layer alias (`self.v`) — set only when legacy mode.
        if not self.is_multi_layer and resolved_indices:
            self.v = self.v_by_layer[resolved_indices[0]]
        else:
            self.v = None

        # Concatenate v_l → flat vector, used for history bookkeeping only.
        flat_v = torch.cat([new_v_by_layer[k] for k in sorted(new_v_by_layer)])
        self.v_history.append(flat_v.clone())
        self.lambda1_history.append(self.lambda_1)
        self.lambda2_history.append(self.lambda_2)
        self.gap_history.append(self.gap)
        self.stability_history.append(stability)
        # Bound history so long-running configurations (>>3 epochs, ablation
        # sweeps that call .update repeatedly) don't accumulate (32*H,)
        # tensors indefinitely. ~50MB per entry at L=32, H=4096, fp32.
        if len(self.v_history) > _MAX_HISTORY:
            drop = len(self.v_history) - _MAX_HISTORY
            self.v_history = self.v_history[drop:]
            self.lambda1_history = self.lambda1_history[drop:]
            self.lambda2_history = self.lambda2_history[drop:]
            self.gap_history = self.gap_history[drop:]
            self.stability_history = self.stability_history[drop:]

        # Theorem 1 verifier reads `last_measurement` after each update().
        # Per-layer rows are flat scalars (JSONL-friendly); anchor vectors
        # are exposed by reference (the verifier dumps them to disk).
        self.last_measurement = {
            "global_step": global_step,
            "epoch": epoch,
            "lr": lr,
            "n_samples_used": int(n_used),
            "per_layer": {
                int(li): {
                    "lambda1": new_l1[li],
                    "lambda2": new_l2[li],
                    "gamma": new_l1[li] - new_l2[li],
                    "delta_sigma_fro": delta_sigma_fro.get(li, float("nan")),
                    "sign_inner_prev": sign_inner_prev.get(li, float("nan")),
                    "d_step": d_step.get(li, float("nan")),
                } for li in (resolved_indices or [])
            },
            # Anchors kept as a separate dict so the verifier can dump npy
            # files step-by-step without re-running PCA.
            "v_by_layer": {int(li): new_v_by_layer[li].clone()
                           for li in (resolved_indices or [])},
        }

        # Restore the model's mode. update() flips to .eval() unconditionally;
        # without this restore a step-level verification call inside the SFT
        # loop would silently leave the model in eval mode for the rest of
        # the epoch (gradient_checkpointing + dropout would also be off).
        if was_training:
            model.train()

        stats = {
            "lambda_1": self.lambda_1,
            "lambda_2": self.lambda_2,
            "gap": self.gap,
            "stability": stability,
            "n_samples_used": int(n_used),
            "num_layers": len(self.layer_indices),
        }
        stab_str = f"{stability:.4f}" if stability == stability else "N/A"
        logger.info(
            "TrajectoryAnchor.update | epoch=%d | layers=%d (%s) | "
            "λ1̄=%.4f | λ2̄=%.4f | gap̄=%.4f | stability=%s | n=%d",
            epoch, len(self.layer_indices),
            f"[{self.layer_indices[0]}..{self.layer_indices[-1]}]"
            if len(self.layer_indices) > 4 else str(self.layer_indices),
            self.lambda_1, self.lambda_2, self.gap, stab_str, n_used,
        )
        return stats

    # ------------------------------------------------------------------ alignment
    @torch.no_grad()
    def compute_alignment(self, states: torch.Tensor) -> torch.Tensor:
        """Compute per-sample alignment score, NAIT Eq 5.

        Accepts either:
          - ``[N, H]`` — legacy single-layer mode; dot with the single v.
          - ``[N, num_layers, H]`` — multi-layer mode; sum ⟨states_l, v_l⟩ over l.

        The result is min-max normalised into [0, 1].
        """
        if not self.is_fitted:
            raise RuntimeError("Anchor not yet fitted. Call update() first.")
        states = states.float().cpu()
        if states.ndim == 2:
            # Single-layer mode: expect exactly one layer in v_by_layer.
            if len(self.layer_indices) != 1:
                raise RuntimeError(
                    f"compute_alignment got 2-D states but anchor was fitted "
                    f"on {len(self.layer_indices)} layers; expected 3-D input."
                )
            v = self.v_by_layer[self.layer_indices[0]].float().cpu()
            alignment = states @ v
        elif states.ndim == 3:
            n, num_layers, _ = states.shape
            if num_layers != len(self.layer_indices):
                raise RuntimeError(
                    f"compute_alignment: states has {num_layers} layers but "
                    f"anchor fitted on {len(self.layer_indices)} layers.",
                )
            # Σ_l ⟨states[:, i, :], v_l⟩  (NAIT Eq 5)
            alignment = torch.zeros(n, dtype=torch.float32)
            for i, li in enumerate(self.layer_indices):
                v_l = self.v_by_layer[li].float().cpu()
                alignment += states[:, i, :] @ v_l
        else:
            raise ValueError(
                f"states must be 2-D or 3-D; got shape {tuple(states.shape)}",
            )
        a_min, a_max = alignment.min(), alignment.max()
        if (a_max - a_min) > 1e-8:
            alignment = (alignment - a_min) / (a_max - a_min)
        else:
            alignment = torch.full_like(alignment, 0.5)
        return alignment

    # ------------------------------------------------------------------ bookkeeping
    def get_history_summary(self) -> Dict[str, list]:
        return {
            "num_epochs_tracked": len(self.v_history),
            "lambda1_per_epoch": self.lambda1_history,
            "lambda2_per_epoch": self.lambda2_history,
            "gap_per_epoch": self.gap_history,
            "stability_per_epoch": self.stability_history,
        }

    def state_dict(self) -> Dict:
        return {
            "v_by_layer": {k: v.cpu() for k, v in self.v_by_layer.items()},
            "layer_indices": self.layer_indices,
            "lambda1_by_layer": self.lambda1_by_layer,
            "lambda2_by_layer": self.lambda2_by_layer,
            "gap_by_layer": self.gap_by_layer,
            "v_history": [v.cpu() for v in self.v_history],
            "lambda1_history": self.lambda1_history,
            "lambda2_history": self.lambda2_history,
            "gap_history": self.gap_history,
            "stability_history": self.stability_history,
            "layer_idx": self.layer_idx,
            "layer_indices_spec": self.layer_indices_spec,
            "max_samples_for_pca": self.max_samples_for_pca,
        }

    def load_state_dict(self, state: Dict) -> None:
        # Multi-layer state.
        self.v_by_layer = {
            int(k): v for k, v in (state.get("v_by_layer") or {}).items()
        }
        self.layer_indices = list(state.get("layer_indices") or [])
        self.lambda1_by_layer = state.get("lambda1_by_layer", {})
        self.lambda2_by_layer = state.get("lambda2_by_layer", {})
        self.gap_by_layer = state.get("gap_by_layer", {})
        # History.
        self.v_history = state.get("v_history", [])
        self.lambda1_history = state.get("lambda1_history", [])
        self.lambda2_history = state.get("lambda2_history", [])
        self.gap_history = state.get("gap_history", [])
        self.stability_history = state.get("stability_history", [])
        # Config.
        self.layer_idx = state.get("layer_idx", -1)
        self.layer_indices_spec = state.get("layer_indices_spec", None)
        self.max_samples_for_pca = state.get("max_samples_for_pca", 2000)
        # Legacy single-layer alias.
        if not self.is_multi_layer and self.layer_indices:
            self.v = self.v_by_layer.get(self.layer_indices[0])
        else:
            self.v = None
        # Legacy single-layer state.
        if "v" in state and state["v"] is not None and not self.v_by_layer:
            # Pre-multi-layer checkpoint — translate.
            v = state["v"]
            self.v = v
            decoder_li = max(0, self.layer_idx if self.layer_idx >= 0 else 0)
            self.v_by_layer = {decoder_li: v}
            self.layer_indices = [decoder_li]
