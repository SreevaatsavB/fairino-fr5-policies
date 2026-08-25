#!/usr/bin/env python3
"""
xr1_stats.py — normalisation statistics for ANY dataset in XR-1's JSON format.

    python xr1_eef/xr1_stats.py <json_dir_or_files...> --out configs/data/<name>.yaml [--batch-size 48]

Reproduces exactly what Xiaomi's `json_dataset.py` computes per sample, so the
statistics describe the tensors their trainer will actually see:

    per arm (left / right):
        Δpos_i  = rotm_t.T @ (actions.ee_pos[t+i] - proprios.ee_pos[t])      tool frame
        Δrot_i  = axis_angle(rotm_t.T @ actions.ee_rotm[t+i])
        Δgrip_i = actions.gripper[t+i] - proprios.gripper[t]
    waist_i     = actions.waist_pos[t+i] - proprios.waist_pos[t]
    base_i      = actions.base_vel[t+i]
    packed to (action_length, 60) at io.ACTION_PARTS, padded by repeating the last
    real entry; every frame t of every episode is a chunk start.

    mean/std   per chunk POSITION, shape (action_length, 60)
    q01/q99    of compose_state(left_gripper, left_joint, right_gripper, right_joint), (1, 60)

Missing arms / waist / base are zeros, which XR-1 treats as padding
(validate_quantiles: q01 == q99 == 0). A state dim that is constant but non-zero
would fail their validator, so it is widened by 1e-3.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

ACTION_LENGTH = 30
ACTION_DIM = STATE_DIM = 60
# io.py ACTION_PARTS, verbatim
PARTS = {"left_ee_pos": slice(0, 3), "left_ee_aa": slice(3, 6), "left_gripper": slice(6, 7),
         "right_ee_pos": slice(8, 11), "right_ee_aa": slice(11, 14), "right_gripper": slice(14, 15),
         "waist": slice(16, 17), "base": slice(17, 20)}


def _arr(d, key, n, width):
    """Nested-key lookup ('proprios.left_ee_pos') -> (n, width) float array, zeros if absent."""
    cur = d
    for k in key.split("."):
        cur = cur.get(k) if isinstance(cur, dict) else None
    if cur is None or len(cur) == 0:
        return np.zeros((n, width))
    a = np.asarray(cur, dtype=np.float64).reshape(n, -1)
    return a


def _pad(x, steps, length):
    return x if steps == length else np.concatenate([x, np.repeat(x[-1:], length - steps, axis=0)])


def episode_relative_actions(ep, length=ACTION_LENGTH):
    """(n, length, 60) packed relative actions for every chunk start of one episode dict."""
    n = int(ep["num_frames"])
    t = np.arange(n)
    idx = np.minimum(t[:, None] + np.arange(length)[None, :], n - 1)      # (n, L) clamped = pad-last
    out = np.zeros((n, length, ACTION_DIM))
    for arm in ("left", "right"):
        p0 = _arr(ep, f"proprios.{arm}_ee_pos", n, 3)
        R0 = _arr(ep, f"proprios.{arm}_ee_rotm", n, 9).reshape(n, 3, 3)
        g0 = _arr(ep, f"proprios.{arm}_gripper_pos", n, 1)
        ap = _arr(ep, f"actions.{arm}_ee_pos", n, 3)
        aR = _arr(ep, f"actions.{arm}_ee_rotm", n, 9).reshape(n, 3, 3)
        ag = _arr(ep, f"actions.{arm}_gripper_pos", n, 1)
        if not np.any(R0):
            continue                                                     # arm absent -> zeros
        R0T = np.transpose(R0, (0, 2, 1))
        dpos = np.einsum("nij,nlj->nli", R0T, ap[idx] - p0[:, None, :])   # (n, L, 3)
        rel = np.einsum("nij,nljk->nlik", R0T, aR[idx])                  # (n, L, 3, 3)
        daa = R.from_matrix(rel.reshape(-1, 3, 3)).as_rotvec().reshape(n, length, 3)
        out[:, :, PARTS[f"{arm}_ee_pos"]] = dpos
        out[:, :, PARTS[f"{arm}_ee_aa"]] = daa
        out[:, :, PARTS[f"{arm}_gripper"]] = ag[idx] - g0[:, None, :]
    w0 = _arr(ep, "proprios.waist_pos", n, 1); aw = _arr(ep, "actions.waist_pos", n, 1)
    out[:, :, PARTS["waist"]] = aw[idx] - w0[:, None, :]
    out[:, :, PARTS["base"]] = _arr(ep, "actions.base_vel", n, 3)[idx]
    return out


def episode_states(ep):
    """(n, 60) packed states, compose_state layout: joints 0-6 / grip 7 / joints 8-14 / grip 15."""
    n = int(ep["num_frames"])
    st = np.zeros((n, STATE_DIM))
    lj = _arr(ep, "proprios.left_arm_joint", n, 0) if ep.get("proprios", {}).get("left_arm_joint") else None
    rj = _arr(ep, "proprios.right_arm_joint", n, 0) if ep.get("proprios", {}).get("right_arm_joint") else None
    if lj is not None:
        assert lj.shape[1] <= 7, "left arm joint state cannot exceed 7"
        st[:, :lj.shape[1]] = lj
    st[:, 7] = _arr(ep, "proprios.left_gripper_pos", n, 1)[:, 0]
    if rj is not None:
        assert rj.shape[1] <= 7, "right arm joint state cannot exceed 7"
        st[:, 8:8 + rj.shape[1]] = rj
    st[:, 15] = _arr(ep, "proprios.right_gripper_pos", n, 1)[:, 0]
    return st


class Stats:
    """Streaming per-position mean/std and state quantiles over many episodes."""

    def __init__(self, length=ACTION_LENGTH):
        self.length, self.n = length, 0
        self.s1 = np.zeros((length, ACTION_DIM)); self.s2 = np.zeros((length, ACTION_DIM))
        self.states = []

    def add(self, ep):
        a = episode_relative_actions(ep, self.length)
        self.s1 += a.sum(0); self.s2 += (a * a).sum(0); self.n += a.shape[0]
        self.states.append(episode_states(ep))

    def result(self):
        mean = self.s1 / max(self.n, 1)
        std = np.sqrt(np.maximum(self.s2 / max(self.n, 1) - mean * mean, 0.0))
        states = np.concatenate(self.states)
        q01 = np.quantile(states, 0.01, axis=0)[None]; q99 = np.quantile(states, 0.99, axis=0)[None]
        tie = (q99 <= q01) & ~((q01 == 0) & (q99 == 0))       # constant non-zero dim: widen
        q99[tie] = q01[tie] + 1e-3
        return mean, std, q01, q99


def write_yaml(path, json_dir, mean, std, q01, q99, batch_size=48, length=ACTION_LENGTH):
    rows = lambda a: "\n".join("      - [" + ", ".join(f"{v:.6g}" for v in r) + "]" for r in np.asarray(a).tolist())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(f"""# @package _global_
# Generated by xr1_eef/xr1_stats.py
data:
  type: BaseDataModule
  params:
    type: json
    max_steps: ${{trainer.max_steps}}
    train_datasets:
      batch_size: {batch_size}
      action_length: {length}
      paths:
      - {Path(json_dir).resolve()}
      mean:
{rows(mean)}
      std:
{rows(std)}
      q01:
{rows(q01)}
      q99:
{rows(q99)}
""")


def find_jsons(paths):
    files = []
    for p in paths:
        p = Path(p)
        files += [p] if p.is_file() else sorted(Path(f) for f in glob.glob(str(p / "**" / "*.json"), recursive=True))
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="XR-1 episode JSON files or directories")
    ap.add_argument("--out", required=True, help="data config yaml to write")
    ap.add_argument("--batch-size", type=int, default=48)
    args = ap.parse_args(argv)
    files = find_jsons(args.paths)
    if not files:
        raise SystemExit("no episode JSON files found")
    st = Stats()
    frames = 0
    for f in files:
        ep = json.loads(Path(f).read_text()); st.add(ep); frames += int(ep["num_frames"])
    mean, std, q01, q99 = st.result()
    json_dir = files[0].parent if len({f.parent for f in files}) == 1 else Path(args.paths[0])
    write_yaml(args.out, json_dir, mean, std, q01, q99, args.batch_size)
    active = [k for k, s in PARTS.items() if std[:, s].max() > 0]
    print(f"{len(files)} episodes, {frames:,} frames -> {st.n:,} chunk starts; active action parts: {active}")
    print(f"  std @ entry 0 / {ACTION_LENGTH-1}: left pos {std[0,0:3].round(5)} / {std[-1,0:3].round(4)}  "
          f"left aa {std[0,3:6].round(5)} / {std[-1,3:6].round(4)}")
    print(f"  state q01/q99 dims with range: {int(((q99 > q01)).sum())} of 60")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
