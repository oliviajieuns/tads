"""Theorem 1 verifier — offline analyser.

Reads
    <run_dir>/thm_verification/metrics.jsonl
    <run_dir>/thm_verification/anchors/step_*.npy

Produces, under <run_dir>/thm_verification/analysis/:
    figures/F1_eigengap_heatmap.{png,pdf}    E1 (assumption A2)
    figures/F2_dSigma_vs_lr.{png,pdf}        E2 (assumption A1) + C_Σ fit
    figures/F3_d_vs_bound.{png,pdf}          E3 (conclusion C1)  ★
    figures/F4_cumulative_S.{png,pdf}        E4 (conclusion C2)
    table_F1_tightness.{csv,tex}             E3 — tightness ratio stats
    summary.json                              §4.5 numbers (γ_min, C_Σ,
                                              bound violations, ratio mean)
    PASS_FAIL.json                            E1..E4 + E3b verdicts

Order:
    E2 (C_Σ regression)  →  E1, E3, E3b, E4  (E3 depends on E2's C_Σ)

Usage:
    python scripts/analyze_theorem.py --run_dir <path>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
except Exception as exc:
    print(f"[fatal] matplotlib import failed: {exc}", file=sys.stderr)
    raise


# ----------------------------------------------------------------- IO
def _load_metrics(metrics_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _by_layer(rows: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(int(r["layer_idx"]), []).append(r)
    for li in out:
        out[li].sort(key=lambda r: int(r["global_step"]))
    return out


# ----------------------------------------------------------------- E2 (C_Σ)
def fit_c_sigma(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Regress ‖ΔΣ‖_F = C_Σ · η_t through the origin, pooled over all
    (layer, step) pairs where ΔΣ is finite."""
    xs, ys = [], []
    for r in rows:
        d = r.get("delta_sigma_fro")
        lr = r.get("lr")
        if d is None or lr is None:
            continue
        if not np.isfinite(d) or not np.isfinite(lr):
            continue
        xs.append(float(lr))
        ys.append(float(d))
    x = np.array(xs); y = np.array(ys)
    if x.size == 0:
        return {"C_sigma": float("nan"), "r2": float("nan"), "n": 0}
    # Through-origin least squares: slope = Σxy / Σx²
    slope = float((x * y).sum() / (x * x).sum())
    y_pred = slope * x
    ss_res = float(((y - y_pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"C_sigma": slope, "r2": r2, "n": int(x.size),
            "x": x, "y": y}


def plot_F2(fit: Dict[str, Any], out_dir: Path) -> None:
    if fit["n"] == 0:
        return
    x = fit["x"]; y = fit["y"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(x, y, s=8, alpha=0.4, label="measurements")
    xs = np.linspace(0, x.max() * 1.05, 50)
    ax.plot(xs, fit["C_sigma"] * xs, "r-",
            label=f"C_Σ={fit['C_sigma']:.3g}, R²={fit['r2']:.3f}")
    ax.set_xlabel(r"$\eta_t$ (learning rate)")
    ax.set_ylabel(r"$\|\Sigma^{(t+1)} - \Sigma^{(t)}\|_F$")
    ax.set_title("E2 — Assumption A1: $\\Sigma$-drift vs lr")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "F2_dSigma_vs_lr.png", dpi=160)
    fig.savefig(out_dir / "F2_dSigma_vs_lr.pdf")
    plt.close(fig)


# ----------------------------------------------------------------- E1 (eigengap)
def plot_F1(per_layer: Dict[int, List[Dict[str, Any]]], out_dir: Path) -> Dict[str, Any]:
    layers = sorted(per_layer.keys())
    if not layers:
        return {"gamma_min": float("nan"), "n_violations": 0}
    steps = sorted({int(r["global_step"]) for r in next(iter(per_layer.values()))})
    M = np.full((len(layers), len(steps)), np.nan)
    step_idx = {s: i for i, s in enumerate(steps)}
    for j, li in enumerate(layers):
        for r in per_layer[li]:
            si = step_idx.get(int(r["global_step"]))
            if si is None:
                continue
            M[j, si] = float(r["gamma"])
    gamma_min = float(np.nanmin(M)) if np.isfinite(M).any() else float("nan")
    n_violations = int((M <= 0).sum())

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("refresh step")
    ax.set_ylabel("decoder layer")
    ax.set_title(f"E1 — γ^(t) per layer (γ_min={gamma_min:.3g}, "
                 f"#γ≤0: {n_violations})")
    fig.colorbar(im, ax=ax, label="γ = λ1 − λ2")
    fig.tight_layout()
    fig.savefig(out_dir / "F1_eigengap_heatmap.png", dpi=160)
    fig.savefig(out_dir / "F1_eigengap_heatmap.pdf")
    plt.close(fig)
    return {"gamma_min": gamma_min, "n_violations": n_violations}


# ----------------------------------------------------------------- E3 (per-step bound)
def analyze_E3(
    per_layer: Dict[int, List[Dict[str, Any]]],
    c_sigma: float,
    gamma_min: float,
    out_dir: Path,
) -> Dict[str, Any]:
    if not np.isfinite(c_sigma) or not np.isfinite(gamma_min) or gamma_min <= 0:
        return {"bound_violations": None, "tightness": None,
                "skipped_reason": "C_sigma or gamma_min invalid"}
    factor = 2.0 * c_sigma / gamma_min
    n_total = 0
    n_violations = 0
    ratios: List[float] = []
    per_layer_curves: Dict[int, Dict[str, np.ndarray]] = {}
    for li, rs in per_layer.items():
        steps, ds, bounds = [], [], []
        for r in rs:
            d = r.get("d_step")
            lr = r.get("lr")
            if d is None or lr is None or not np.isfinite(d) or not np.isfinite(lr):
                continue
            b = factor * float(lr)
            steps.append(int(r["global_step"]))
            ds.append(float(d))
            bounds.append(b)
            n_total += 1
            if d > b:
                n_violations += 1
            if b > 0:
                ratios.append(d / b)
        per_layer_curves[li] = {
            "steps": np.array(steps),
            "d": np.array(ds),
            "bound": np.array(bounds),
        }

    # F3: per-layer curves (sample a few layers to keep the plot readable).
    layers = sorted(per_layer_curves.keys())
    sample = layers if len(layers) <= 6 else [
        layers[0],
        layers[len(layers) // 4],
        layers[len(layers) // 2],
        layers[3 * len(layers) // 4],
        layers[-1],
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    for li in sample:
        c = per_layer_curves[li]
        if c["steps"].size == 0:
            continue
        line, = ax.plot(c["steps"], c["d"], "-", label=f"L{li} d")
        ax.plot(c["steps"], c["bound"], "--", color=line.get_color(),
                alpha=0.5, label=f"L{li} bound")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel(r"$\|v^{(t+1)} - v^{(t)}\|$")
    ax.set_title(f"E3 — per-step bound | violations: {n_violations}/{n_total}")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "F3_d_vs_bound.png", dpi=160)
    fig.savefig(out_dir / "F3_d_vs_bound.pdf")
    plt.close(fig)

    ratios_arr = np.array(ratios) if ratios else np.array([np.nan])
    tightness = {
        "mean": float(np.nanmean(ratios_arr)),
        "median": float(np.nanmedian(ratios_arr)),
        "min": float(np.nanmin(ratios_arr)),
        "max": float(np.nanmax(ratios_arr)),
        "p5": float(np.nanpercentile(ratios_arr, 5)),
        "p95": float(np.nanpercentile(ratios_arr, 95)),
        "n": int(np.isfinite(ratios_arr).sum()),
    }
    # Table F1 — tightness by layer group.
    layer_groups = _group_layers(layers)
    table_rows: List[Dict[str, Any]] = []
    for grp_name, grp_layers in layer_groups.items():
        rs: List[float] = []
        for li in grp_layers:
            for r in per_layer[li]:
                d = r.get("d_step"); lr = r.get("lr")
                if d is None or lr is None:
                    continue
                if not (np.isfinite(d) and np.isfinite(lr)):
                    continue
                b = factor * float(lr)
                if b > 0 and np.isfinite(d):
                    rs.append(d / b)
        if rs:
            arr = np.array(rs)
            table_rows.append({
                "group": grp_name,
                "n": int(arr.size),
                "min": float(arr.min()),
                "mean": float(arr.mean()),
                "max": float(arr.max()),
            })
    _write_table(table_rows, out_dir, "table_F1_tightness")

    return {
        "factor_2CoverG": factor,
        "n_measurements": n_total,
        "bound_violations": n_violations,
        "tightness": tightness,
        "table_F1": table_rows,
    }


def _group_layers(layers: List[int]) -> Dict[str, List[int]]:
    if not layers:
        return {}
    n = len(layers)
    return {
        "early": layers[: max(1, n // 3)],
        "mid": layers[max(1, n // 3): max(2, 2 * n // 3)],
        "late": layers[max(2, 2 * n // 3):],
    }


def _write_table(rows: List[Dict[str, Any]], out_dir: Path, stem: str) -> None:
    csv_path = out_dir.parent / f"{stem}.csv"
    tex_path = out_dir.parent / f"{stem}.tex"
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(csv_path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    # LaTeX (booktabs)
    with open(tex_path, "w") as f:
        f.write("\\begin{tabular}{l" + "r" * (len(keys) - 1) + "}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(keys) + " \\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            cells = []
            for k in keys:
                v = r[k]
                if isinstance(v, float):
                    cells.append(f"{v:.3g}")
                else:
                    cells.append(str(v))
            f.write(" & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


# ----------------------------------------------------------------- E3b (sign flip)
def analyze_E3b(per_layer: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
    n_flips = 0
    n_checked = 0
    for li, rs in per_layer.items():
        for r in rs:
            s = r.get("sign_inner_prev")
            if s is None or not np.isfinite(s):
                continue
            n_checked += 1
            if s < 0:
                n_flips += 1
    return {"n_sign_checks": n_checked, "n_sign_flips": n_flips}


# ----------------------------------------------------------------- E4 (cumulative)
def plot_F4(
    per_layer: Dict[int, List[Dict[str, Any]]],
    out_dir: Path,
) -> Dict[str, Any]:
    layers = sorted(per_layer.keys())
    if not layers:
        return {"monotone_fraction": float("nan")}
    fig, ax = plt.subplots(figsize=(6, 4))
    n_monotone = 0
    n_total = 0
    sample = layers if len(layers) <= 4 else [
        layers[0], layers[len(layers) // 2], layers[-1],
    ]
    # lr-tail reference: Σ_{s≥t} η_s. Use the first layer's row order
    # (every layer shares the same step / lr schedule).
    rs0 = per_layer[layers[0]]
    steps = [int(r["global_step"]) for r in rs0]
    lrs = [float(r["lr"]) for r in rs0 if r.get("lr") is not None]
    lr_tail = np.cumsum(lrs[::-1])[::-1] if lrs else np.array([])

    for li in sample:
        rs = per_layer[li]
        ds = [r.get("d_step") for r in rs]
        ds = [float(x) if x is not None and np.isfinite(x) else 0.0 for x in ds]
        S = np.cumsum(ds[::-1])[::-1]
        ax.plot(steps[: S.size], S, label=f"L{li}")
    for li in layers:
        rs = per_layer[li]
        ds = [r.get("d_step") for r in rs]
        ds = [float(x) if x is not None and np.isfinite(x) else 0.0 for x in ds]
        S = np.cumsum(ds[::-1])[::-1]
        n_total += 1
        # monotone non-increasing check (allow tiny rounding slack)
        if S.size > 1 and np.all(np.diff(S) <= 1e-9):
            n_monotone += 1
    if lr_tail.size:
        # Rescale so it overlays nicely
        scale = (ax.get_ylim()[1] / (lr_tail.max() + 1e-12))
        ax.plot(steps[: lr_tail.size], lr_tail * scale, "k--",
                label="Σ η_s (rescaled)", alpha=0.6)
    ax.set_xlabel("optimizer step t")
    ax.set_ylabel(r"$S_l(t) = \sum_{s \geq t} d_l^{(s)}$")
    ax.set_title(f"E4 — cumulative tail | monotone: "
                 f"{n_monotone}/{n_total}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "F4_cumulative_S.png", dpi=160)
    fig.savefig(out_dir / "F4_cumulative_S.pdf")
    plt.close(fig)
    return {
        "n_layers": n_total,
        "n_monotone": n_monotone,
        "monotone_fraction": (n_monotone / n_total) if n_total else float("nan"),
    }


# ----------------------------------------------------------------- driver
def run(run_dir: Path) -> Dict[str, Any]:
    ver_dir = run_dir / "thm_verification"
    metrics_path = ver_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"No metrics.jsonl under {ver_dir}. "
            "Did the run have `verification.enabled: true`?"
        )
    out_dir = ver_dir / "analysis"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_metrics(metrics_path)
    per_layer = _by_layer(rows)

    # E2 first — its C_Σ feeds E3.
    e2 = fit_c_sigma(rows)
    plot_F2(e2, fig_dir)

    e1 = plot_F1(per_layer, fig_dir)
    e3 = analyze_E3(per_layer, c_sigma=e2["C_sigma"],
                    gamma_min=e1["gamma_min"], out_dir=fig_dir)
    e3b = analyze_E3b(per_layer)
    e4 = plot_F4(per_layer, fig_dir)

    verdicts = _verdicts(e1, e2, e3, e3b, e4)
    summary = {
        "E1_eigengap": e1,
        "E2_C_sigma": {k: e2[k] for k in ("C_sigma", "r2", "n")},
        "E3_per_step_bound": {k: e3.get(k) for k in
                              ("factor_2CoverG", "n_measurements",
                               "bound_violations", "tightness", "table_F1")},
        "E3b_sign_calibration": e3b,
        "E4_cumulative": e4,
        "verdicts": verdicts,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "PASS_FAIL.json").write_text(json.dumps(verdicts, indent=2))
    print(json.dumps(verdicts, indent=2))
    return summary


def _verdicts(e1, e2, e3, e3b, e4) -> Dict[str, str]:
    v: Dict[str, str] = {}
    v["E1"] = "PASS" if (np.isfinite(e1.get("gamma_min", float("nan")))
                         and e1["gamma_min"] > 0
                         and e1["n_violations"] == 0) else "FAIL"
    v["E2"] = "PASS" if (np.isfinite(e2.get("C_sigma", float("nan")))
                         and np.isfinite(e2.get("r2", float("nan")))
                         and e2["r2"] > 0.3
                         and e2["n"] > 0) else "FAIL"
    bv = e3.get("bound_violations")
    tight = (e3.get("tightness") or {}).get("mean")
    v["E3"] = "SKIPPED" if bv is None else (
        "PASS" if (bv == 0 and tight is not None and 0.001 < tight < 1.0)
        else ("VACUOUS" if (bv == 0 and tight is not None and tight <= 0.001)
              else "FAIL")
    )
    v["E3b"] = "PASS" if e3b.get("n_sign_flips", 0) == 0 else "FAIL"
    mf = e4.get("monotone_fraction")
    v["E4"] = "PASS" if (mf is not None and np.isfinite(mf) and mf >= 0.8) else "FAIL"
    return v


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, type=Path,
                   help="Path to <output_root>/<output_subdir>/runs/<tag>/")
    args = p.parse_args()
    run(args.run_dir)


if __name__ == "__main__":
    main()
