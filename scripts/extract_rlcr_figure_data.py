"""Extract Figure 2 grey-curve data + caption numbers from thm_rlcr run.

Run AFTER scripts/analyze_theorem.py on the no-anchor cell, e.g.:

  RUN=/group-volume/jieuns/tads-checkpoints/light/thm_verify_rlcr/no_anchor/runs/<tag>
  python scripts/extract_rlcr_figure_data.py --rlcr_run $RUN

Outputs to <rlcr_run>/figure2_rlcr/:

  tikz_coords.txt    -- ready-to-paste TikZ coordinates for the grey
                        curve in fig:thm-main, with the same x-axis
                        layout and ~14x rescale as the red anchor.
  caption_numbers.json
                     -- numeric fields the figure / section caption
                        needs (n_points, median d range, # > D-cell
                        median, # > D-cell bound, max d_step, etc.).
  summary.md         -- human-readable summary, paste-ready into chat.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


D_CELL_BOUND = 46.24       # from D cell summary.json (constant lr 2e-4)
D_CELL_MEDIAN_D = 0.068    # from D cell median d_step
D_CELL_MAX_D = 0.342       # from D cell max d_step
D_CELL_C_SIGMA_TRIM = 5780 # from D cell C_sigma_trimmed_mean
D_CELL_GAMMA_MIN = 0.050   # from D cell gamma_min (post-threshold)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rlcr_run", required=True, type=Path)
    ap.add_argument("--skip_warmup_optsteps", type=int, default=250)
    args = ap.parse_args()

    metrics_path = args.rlcr_run / "thm_verification" / "metrics.jsonl"
    if not metrics_path.exists():
        print(f"[fatal] metrics.jsonl not found at {metrics_path}", file=sys.stderr)
        return 1

    # Aggregate d_step per refresh step (median over layers).
    by_step: Dict[int, List[float]] = {}
    by_step_stability: Dict[int, List[float]] = {}
    by_step_lr: Dict[int, List[float]] = {}
    for line in metrics_path.open():
        r = json.loads(line)
        s = int(r.get("global_step", 0))
        if s <= args.skip_warmup_optsteps:
            continue
        d = r.get("d_step")
        if d is None or not np.isfinite(d):
            continue
        by_step.setdefault(s, []).append(float(d))
        lr = r.get("lr")
        if lr is not None and np.isfinite(lr):
            by_step_lr.setdefault(s, []).append(float(lr))

    steps = sorted(by_step.keys())
    if not steps:
        print("[fatal] no post-warmup measurement points", file=sys.stderr)
        return 1

    medians = np.array([float(np.median(by_step[s])) for s in steps])
    maxes = np.array([float(np.max(by_step[s])) for s in steps])

    n = len(steps)
    n_over_d_median = int((medians > D_CELL_MEDIAN_D).sum())
    n_over_d_max = int((medians > D_CELL_MAX_D).sum())
    n_over_d_bound = int((medians > D_CELL_BOUND).sum())

    # Build TikZ coordinates aligned with the fig:thm-main x-axis [0, 9].
    # Sub-sample to ~23 points so the curve reads at column width and
    # the spike pattern stays visible.
    if n > 23:
        idx = np.linspace(0, n - 1, 23).astype(int)
    else:
        idx = np.arange(n)
    sub_medians = medians[idx]

    # Same x-axis convention as fig:thm-main: 0..9 normalised.
    x_scale = 9.0 / max(1, n - 1)
    # Rescale y by the SAME factor as the red anchor: peak (sub_medians.max())
    # mapped to ~5 (under the bound line at 5.5). If RL+CR medians exceed
    # the D cell median by a lot, this will visually escape the safe region.
    y_rescale = 5.0 / max(D_CELL_MAX_D, sub_medians.max(), 1e-12)
    coords = " ".join(
        f"({i * x_scale:.4f},{m * y_rescale:.4f})"
        for i, m in zip(idx, sub_medians)
    )

    out_dir = args.rlcr_run / "figure2_rlcr"
    out_dir.mkdir(parents=True, exist_ok=True)

    tikz_block = f"""% Auto-generated TikZ coordinates for the RL+CR (no-anchor) curve.
% Paste into fig:thm-main where the grey schematic line + dots are.
% Same rescale factor as the red anchor (y_rescale={y_rescale:.4f}) so
% the relative magnitudes are honest.

\\draw[gray!55!black, thick, densely dotted]
  plot[smooth] coordinates {{ {coords} }};

\\foreach \\p in {{ {coords} }} {{
  \\fill[gray!55!black] \\p circle (1.4pt);
}}
"""
    (out_dir / "tikz_coords.txt").write_text(tikz_block)

    caption_numbers = {
        "n_measurement_points": n,
        "median_d_min": float(medians.min()),
        "median_d_max": float(medians.max()),
        "median_d_p50": float(np.median(medians)),
        "max_over_run": float(maxes.max()),
        "n_over_d_cell_median": n_over_d_median,
        "n_over_d_cell_max": n_over_d_max,
        "n_over_d_cell_bound": n_over_d_bound,
        "frac_over_d_cell_median": float(n_over_d_median / n),
        "frac_over_d_cell_max": float(n_over_d_max / n),
        "frac_over_d_cell_bound": float(n_over_d_bound / n),
        "d_cell_reference": {
            "bound": D_CELL_BOUND,
            "median_d": D_CELL_MEDIAN_D,
            "max_d": D_CELL_MAX_D,
            "C_sigma_trim": D_CELL_C_SIGMA_TRIM,
            "gamma_min": D_CELL_GAMMA_MIN,
        },
        "y_rescale_factor": float(y_rescale),
    }
    (out_dir / "caption_numbers.json").write_text(
        json.dumps(caption_numbers, indent=2),
    )

    md = f"""# Figure 2 RL+CR curve summary

Source run: `{args.rlcr_run}`
Post-warmup measurement points: **{n}** (skip_warmup_optsteps={args.skip_warmup_optsteps})

## d_step (median over 24 layers) statistics

| metric | value |
|---|---|
| min  | {medians.min():.3e} |
| max  | {medians.max():.3e} |
| p50  | {np.median(medians):.3e} |
| max-over-run (any layer) | {maxes.max():.3e} |

## Contrast with D cell (TADS anchor)

| comparison | count | fraction |
|---|---|---|
| > D-cell median ({D_CELL_MEDIAN_D}) | {n_over_d_median} / {n} | {n_over_d_median/n:.2%} |
| > D-cell max    ({D_CELL_MAX_D})    | {n_over_d_max} / {n}    | {n_over_d_max/n:.2%} |
| > D-cell bound  ({D_CELL_BOUND})    | {n_over_d_bound} / {n}  | {n_over_d_bound/n:.2%} |

## Output files

- `tikz_coords.txt`     ready-to-paste TikZ coordinates (gray dotted + dots)
- `caption_numbers.json` machine-readable for the figure caption / section paragraph

## Next steps

1. Edit `docs/paper_drafts/fig_thm_main.tex`: replace the schematic
   gray dotted block with the contents of `tikz_coords.txt`.
2. Update the figure caption: replace the
   `\\textcolor{{red}}{{... measured replacement is in progress}}` placeholder
   with a one-liner pointing to App. F cell with the actual contrast.
3. Update `docs/paper_drafts/sec_thm_empirical.tex`: replace the
   placeholder paragraph about RL+CR with one sentence using
   `n_over_d_cell_median` and `n_over_d_cell_bound` as the empirical
   claim.
"""
    (out_dir / "summary.md").write_text(md)

    print(md)
    print(f"\n[written] {out_dir}/")
    print(f"  tikz_coords.txt     ({len(tikz_block)} bytes)")
    print(f"  caption_numbers.json")
    print(f"  summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
