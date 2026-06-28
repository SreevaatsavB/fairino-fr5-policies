#!/usr/bin/env python3
"""
plot_metrics.py — plot the training curves (KL + losses) from a run's metrics.csv.

The CVAE KL fix succeeds iff the KL **stabilises at a non-trivial value (~0.5-5)** instead of
cratering to ~0 (posterior collapse — see docs/failure_analysis/06,10). This makes that visible.

Usage:
    python tools/plot_metrics.py policies/act/checkpoints_joint_dino/metrics.csv
    python tools/plot_metrics.py <ckpt_dir>            # finds metrics.csv inside

Writes a PNG next to the CSV and prints a quick verdict on whether the latent stayed alive.
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python tools/plot_metrics.py <metrics.csv | checkpoint_dir>")
    p = Path(sys.argv[1])
    path = p / "metrics.csv" if p.is_dir() else p
    if not path.exists():
        sys.exit(f"no metrics file at {path} (train first — train.py writes it per epoch)")

    rows = list(csv.DictReader(open(path)))
    if not rows:
        sys.exit(f"{path} is empty")
    ep = [int(r["epoch"]) for r in rows]

    def col(k):
        return [float(r[k]) if r.get(k) not in (None, "") else float("nan") for r in rows]

    train_l1, val_l1, train_kl, kl_w = col("train_l1"), col("val_l1"), col("train_kl"), col("kl_weight_eff")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    # left: reconstruction loss (train vs val) — gap = overfitting (see doc 09)
    ax1.plot(ep, train_l1, label="train L1")
    ax1.plot(ep, val_l1, label="val L1")
    ax1.set_title("reconstruction loss"); ax1.set_xlabel("epoch"); ax1.set_ylabel("L1")
    ax1.legend(); ax1.grid(alpha=.3)

    # right: CVAE KL — should stabilise in the healthy band, not collapse
    ax2.plot(ep, train_kl, color="tab:red", label="train KL")
    ax2.axhspan(0.5, 5, color="green", alpha=.12, label="healthy band (0.5-5)")
    if any(k == k for k in kl_w):                       # overlay the annealed weight if present
        ax2b = ax2.twinx()
        ax2b.plot(ep, kl_w, color="tab:gray", ls="--", lw=1, label="kl_weight (annealed)")
        ax2b.set_ylabel("kl_weight", color="tab:gray")
    ax2.set_title("CVAE KL  (target: stabilise, NOT -> 0)")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("KL (nats)"); ax2.legend(loc="upper right"); ax2.grid(alpha=.3)

    out = path.with_suffix(".png")
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"wrote {out}")

    # verdict
    kl = [k for k in train_kl if k == k]
    if kl:
        final = sum(kl[-3:]) / len(kl[-3:])
        gap = (val_l1[-1] - train_l1[-1]) if (val_l1[-1] == val_l1[-1]) else float("nan")
        verdict = ("HEALTHY — latent alive" if final >= 0.1
                   else "COLLAPSED — latent dead (KL ~ 0); see docs/failure_analysis/06,10")
        print(f"final KL (last 3 epochs) ~ {final:.4f}  ->  {verdict}")
        print(f"final train/val L1 = {train_l1[-1]:.4f} / {val_l1[-1]:.4f}  (gap {gap:.4f}; large gap = overfitting, doc 09)")


if __name__ == "__main__":
    main()
