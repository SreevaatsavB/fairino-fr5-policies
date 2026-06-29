#!/usr/bin/env python3
"""
plot_actions.py — plot a saved action record (from common/action_record.py) as one
subplot per action dimension, over time.

Works for any record produced by the shared layer:
  • offline eval (experiments/eval_on_test.py)  -> has ground truth -> GT vs prediction
  • live robot rollout (common/deploy.py)        -> no ground truth  -> prediction only

Usage:
    python tools/plot_actions.py policies/act/checkpoints/eval/ep000.npz
    python tools/plot_actions.py <record.npz> --out my_plot.png

Prints a per-dimension MAE table when ground truth is present.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "common"))
from action_record import load_record, dim_labels   # noqa: E402


def plot_record(path, out=None):
    arrays, meta = load_record(path)
    pred = arrays.get("pred")
    if pred is None:
        sys.exit(f"{path} has no 'pred' array — nothing to plot")
    gt = arrays.get("gt")
    t  = arrays.get("t", np.arange(len(pred)))

    action_dim = pred.shape[1]
    labels = meta.get("dim_labels") or dim_labels(meta.get("action_space", ""), action_dim)
    has_gt = gt is not None and np.isfinite(gt).any()

    # grid: ~2 columns, one subplot per action dim
    ncol = 2 if action_dim > 1 else 1
    nrow = int(np.ceil(action_dim / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 2.1 * nrow),
                             squeeze=False, sharex=True)
    axes = axes.ravel()

    mae = []
    for i in range(action_dim):
        ax = axes[i]
        if has_gt:
            ax.plot(t, gt[:, i], color="tab:green", lw=1.6, label="ground truth")
            ax.plot(t, pred[:, i], color="tab:red", lw=1.3, ls="--", label="prediction")
            di = np.abs(pred[:, i] - gt[:, i])
            m = float(np.nanmean(di)); mae.append(m)
            ax.set_title(f"{labels[i]}   (MAE {m:.3f})", fontsize=10)
        else:
            ax.plot(t, pred[:, i], color="tab:red", lw=1.3, label="prediction")
            ax.set_title(labels[i], fontsize=10)
        ax.grid(alpha=.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    for j in range(action_dim, len(axes)):   # hide unused cells
        axes[j].axis("off")
    for ax in axes[-ncol:]:
        ax.set_xlabel("timestep")

    title = (f"{meta.get('policy','?')} / {meta.get('source','?')} — "
             f"actions over time ({meta.get('action_space','?')})")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out = Path(out) if out else Path(path).with_suffix(".png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")

    if has_gt and mae:
        print("\nper-dimension MAE (prediction vs ground truth):")
        for lab, m in zip(labels, mae):
            print(f"  {lab:>8}: {m:.4f}")
        print(f"  {'overall':>8}: {np.mean(mae):.4f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("record", help="path to a .npz saved by ActionRecorder")
    ap.add_argument("--out", default=None, help="output PNG (default: alongside the npz)")
    args = ap.parse_args()
    plot_record(args.record, args.out)


if __name__ == "__main__":
    main()
