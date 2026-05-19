"""Compute cumulative anchor drift vs Theorem 1 tail bound.

For each refresh step t, compute:
  measured_drift(t) = median over layers of ||v_l^(t) - v_l^(T)||  (T = final step)
  bound(t)          = (2 * C_Sigma / gamma_min) * sum_{s >= t} eta_s

Both curves -> 0 as t -> T under a decaying learning rate schedule.
Outputs TikZ coordinates ready for fig:thm-main.

Usage:
  python scripts/extract_thm_main_cumulative.py \
      --run /group-volume/jieuns/tads-checkpoints/light/thm_verify_ablation/B_probe2000_cosine/runs/<tag>
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--gamma_min", type=float, default=0.050)
    ap.add_argument("--c_sigma_trimmed", type=float, default=5.78e3)
    ap.add_argument("--skip_warmup_optsteps", type=int, default=0,
                    help="cosine schedule has natural warmup; keep all by default")
    args = ap.parse_args()

    anc_dir = args.run / "thm_verification" / "anchors"
    met_path = args.run / "thm_verification" / "metrics.jsonl"
    if not anc_dir.exists() or not met_path.exists():
        print(f"[fatal] missing anchors/ or metrics.jsonl in {args.run}", file=sys.stderr)
        return 1

    # 1) Load anchors. step_<global_step>.npy contains shape (L, d).
    anchor_files = sorted(anc_dir.glob("step_*.npy"))
    steps = [int(f.stem.split("_")[1]) for f in anchor_files]
    anchors = np.stack([np.load(f) for f in anchor_files])
    print(f"[load] {len(steps)} steps, layers={anchors.shape[1]}, hidden={anchors.shape[2]}")
    print(f"       first step={steps[0]}, last step={steps[-1]}")

    # 2) Sign-calibrate per layer relative to the previous timestep.
    L = anchors.shape[1]
    for li in range(L):
        for ti in range(1, len(steps)):
            if np.dot(anchors[ti, li], anchors[ti - 1, li]) < 0:
                anchors[ti, li] = -anchors[ti, li]

    # 3) Measured cumulative drift to final.
    v_final = anchors[-1]
    diffs = anchors - v_final[None, :, :]
    d_to_T = np.linalg.norm(diffs, axis=2)
    median_d = np.median(d_to_T, axis=1)
    p95_d = np.percentile(d_to_T, 95, axis=1)

    # 4) Per-step learning rate (median across layers at each refresh).
    lr_by_step = defaultdict(list)
    for line in met_path.open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        s = int(r["global_step"])
        lr_by_step[s].append(float(r["lr"]))
    step_lr = {s: float(np.median(v)) for s, v in lr_by_step.items() if v}

    # 5) Theorem 1 cumulative tail bound at each refresh.
    refresh_interval = steps[1] - steps[0] if len(steps) > 1 else 25
    eta_at_refresh = np.array([step_lr.get(s, np.nan) for s in steps])
    tail_eta = np.zeros_like(eta_at_refresh)
    cum = 0.0
    for i in range(len(steps) - 2, -1, -1):
        cum += eta_at_refresh[i] * refresh_interval
        tail_eta[i] = cum
    bound = (2.0 * args.c_sigma_trimmed / args.gamma_min) * tail_eta

    # 6) Print summary table.
    print("\nstep | lr        | tail_sum_eta | bound       | median_d   | p95_d")
    print("-" * 78)
    for i, s in enumerate(steps):
        print(f"{s:5d} | {eta_at_refresh[i]:.3e} | {tail_eta[i]:.3e}   | "
              f"{bound[i]:.3e} | {median_d[i]:.3e} | {p95_d[i]:.3e}")

    # 7) Paper-side outputs.
    out_dir = args.run / "thm_verification" / "cumulative_figure"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Log-space plot coordinates (matches the hand-tuned figure in
    # paper_v4/sections/04_experiments.tex). Sub-sample to ~20 points.
    n = len(steps)
    idx = np.linspace(0, n - 1, 20).astype(int)
    x_min, x_max = steps[0], steps[-1]
    def xc(s): return 9.0 * (s - x_min) / max(1, x_max - x_min)
    def yc_log(v):
        v = max(float(v), 1e-12)
        return (np.log10(v) + 3.0) * 0.75  # log10 axis [-3, +5] -> [0, 6]

    bound_coords = " ".join(
        f"({xc(steps[i]):.3f},{yc_log(bound[i]):.3f})" for i in idx)
    drift_coords = " ".join(
        f"({xc(steps[i]):.3f},{yc_log(median_d[i]):.3f})" for i in idx)

    tikz = (
        "% Cumulative drift to final anchor (red) and Theorem 1 tail bound (blue dashed).\n"
        "% Both curves on log10 y-axis: y_plot = (log10(value) + 3) * 0.75, range [-3, +5].\n\n"
        "\\draw[blue!70!black, very thick, dashed]\n"
        f"  plot[smooth] coordinates {{ {bound_coords} }};\n\n"
        "\\draw[red!75!black, thick]\n"
        f"  plot[smooth] coordinates {{ {drift_coords} }};\n"
        f"\\foreach \\p in {{ {drift_coords} }} {{\n"
        "  \\fill[red!75!black] \\p circle (1.5pt);\n"
        "}\n"
    )
    (out_dir / "tikz_cumulative.txt").write_text(tikz)

    summary = {
        "n_refresh_points": len(steps),
        "first_step": int(steps[0]),
        "last_step": int(steps[-1]),
        "C_sigma_trimmed": float(args.c_sigma_trimmed),
        "gamma_min": float(args.gamma_min),
        "bound_initial": float(bound[0]),
        "bound_final": float(bound[-1]),
        "median_d_initial": float(median_d[0]),
        "median_d_final": float(median_d[-1]),
    }
    (out_dir / "cumulative_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n[written] {out_dir}/tikz_cumulative.txt")
    print(f"[written] {out_dir}/cumulative_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
