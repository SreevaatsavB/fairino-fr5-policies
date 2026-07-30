"""
gate.py — offline pass/fail before booking robot time. Runs on the Linux GPU PC
(or anywhere) against the npz files run.py eval already writes:

    python delta_joint/run.py eval --ckpt best.pt --episodes 4
    python delta_joint/gate.py <ckpt_dir>/eval

Pre-registered criteria — written down BEFORE the run so nobody talks themselves
into "close enough" after 27 GPU-hours (README §6):

  1. model MAE < the "don't move" baseline on every episode. The 30k
     absolute-action run was 22x WORSE than not moving; a policy that loses to a
     frozen arm cannot control one.
  2. the predicted gripper crosses the 0.65 close threshold in at least half the
     episodes whose ground truth closes. The 30k run peaked at 0.154.

The mean-action baseline is printed for context (model << mean-action means it
learned the trajectory; the 30k run already passed that and still failed, which
is why it is not a criterion). val_l1 is deliberately not consulted: it is
normalized-space MSE and it scored 0.0339 on the run that lost to "don't move".

Exit code 0 = criteria met, book robot time. 1 = keep training.
"""

import sys
from glob import glob
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from action_record import load_record  # noqa: E402

CLOSE = 0.65   # gripper close threshold — same as deploy's GRIPPER_CLOSE_THRESH


def episode_stats(arrays):
    """Model vs the two trivial baselines, plus gripper commitment, for one npz."""
    pred, gt, state = arrays["pred"], arrays["gt"], arrays["state"]
    ok = ~np.isnan(gt[:, :6]).any(1)
    pred, gt, state = pred[ok], gt[ok], state[ok]
    return {
        "n":         int(ok.sum()),
        "model":     float(np.abs(pred[:, :6] - gt[:, :6]).mean()),
        "null":      float(np.abs(state[:, :6] - gt[:, :6]).mean()),   # "don't move"
        "mean_act":  float(np.abs(gt[:, :6] - gt[:, :6].mean(0)).mean()),
        "gt_closes":   bool((gt[:, 6] >= CLOSE).any()),
        "pred_closes": bool((pred[:, 6] >= CLOSE).any()),
        "pred_grip_max": float(pred[:, 6].max()),
    }


def verdict(stats):
    """(passed: bool, lines: list[str]) from per-episode stats."""
    lines, beats_null = [], []
    for name, s in stats:
        beat = s["model"] < s["null"]
        beats_null.append(beat)
        lines.append(
            f"  {name}: n={s['n']}  model {s['model']:.3f}  "
            f"dont-move {s['null']:.3f}  mean-action {s['mean_act']:.3f} deg  "
            f"{'BEATS null' if beat else '** LOSES to null **'}")

    closers = [s for _, s in stats if s["gt_closes"]]
    hits = sum(s["pred_closes"] for s in closers)
    a_pass = all(beats_null)
    b_pass = (not closers) or hits >= (len(closers) + 1) // 2
    lines.append(f"  1. model beats don't-move on every episode: "
                 f"{sum(beats_null)}/{len(beats_null)}  "
                 f"{'PASS' if a_pass else 'FAIL'}")
    grip_max = max((s["pred_grip_max"] for _, s in stats), default=0.0)
    lines.append(f"  2. gripper crosses {CLOSE} where GT closes: "
                 f"{hits}/{len(closers)} episodes (pred max {grip_max:.3f})  "
                 f"{'PASS' if b_pass else 'FAIL'}"
                 + ("  [no GT closes found — criterion vacuous, check episodes]"
                    if not closers else ""))
    return a_pass and b_pass, lines


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: gate.py <eval_dir with ep*.npz>\n{__doc__}")
    files = sorted(glob(str(Path(sys.argv[1]) / "ep*.npz")))
    if not files:
        raise SystemExit(f"no ep*.npz in {sys.argv[1]} — run "
                         f"`python delta_joint/run.py eval --ckpt <best.pt>` first")
    stats = [(Path(f).stem, episode_stats(load_record(f)[0])) for f in files]
    passed, lines = verdict(stats)
    print("\n".join(lines))
    print(f"\nVERDICT: {'DEPLOYABLE — book robot time' if passed else 'NOT DEPLOYABLE — keep training'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
