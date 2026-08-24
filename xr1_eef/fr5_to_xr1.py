#!/usr/bin/env python3
"""
fr5_to_xr1.py — convert the FR5 LeRobot-v2 dataset to Xiaomi-Robotics-1 format.

    python xr1_eef/fr5_to_xr1.py <lerobot_root> --out <dir> [--dry-run] [--episodes 0 1 2]

Writes  <dir>/json/episode_XXX.json      one per episode (XR-1 schema, docs/data_format.md)
        <dir>/configs/fr5.yaml           XR-1 data config with the (30, 60) per-step
                                         mean/std of the packed RELATIVE actions and the
                                         (1, 60) q01/q99 of the packed state
Videos are referenced in place (absolute paths), not copied.

Conventions (every one verified, see README §5):
  FR5 eef_pose  = [x, y, z] mm + [rx, ry, rz] deg, EXTRINSIC XYZ (Fairino robot_types.h:
                  "Rotation Angle about fixed axis X/Y/Z") -> R = Rz·Ry·Rx = scipy 'xyz'
  XR-1 wants    metres, radians, flattened 3x3 rotm, joints in radians
  action[t]     = the pose commanded for frame t. We have observed poses only, so
                  action[t] := proprio[t+1] (last frame repeats) — matches their demo,
                  where action[t] ≈ proprio[t] + one step.
  single arm    -> XR-1's LEFT slot; right arm / waist / base are zeros, which
                  validate_quantiles treats as padding (q01 == q99 == 0).
  stats         computed with XR-1's own delta formulas (json_dataset._arm_action),
                  per chunk position, over every (episode, frame) start — exactly the
                  samples the trainer will draw.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

ACTION_LENGTH = 30
ACTION_DIM = STATE_DIM = 60
# XR-1 io.py ACTION_PARTS, left arm only
SL_POS, SL_AA, SL_GRIP = slice(0, 3), slice(3, 6), slice(6, 7)

# Fixed rotation applied to every FR5 tool orientation so that our TCP frame matches
# XR-1's canonical end-effector frame ("we unify the orientation of the end-effector
# frames across all robot data and UMI data"). Their canonical axes are undocumented;
# identity until determined. README §6 risk 1.
TOOL_FRAME_FIX = np.eye(3)

CAMS = {"ego": "observation.images.scene_cam", "wrist_left": "observation.images.wrist_cam"}


def rpy_deg_to_rotm(rpy_deg):
    """(N,3) Fairino [rx,ry,rz] deg -> (N,3,3). scipy lowercase 'xyz' = extrinsic."""
    return R.from_euler("xyz", np.asarray(rpy_deg, dtype=np.float64), degrees=True).as_matrix() @ TOOL_FRAME_FIX


def rotm_to_rpy_deg(rotm):
    """Inverse of rpy_deg_to_rotm, for deploy. (N,3,3) -> (N,3) deg."""
    return R.from_matrix(np.asarray(rotm) @ TOOL_FRAME_FIX.T).as_euler("xyz", degrees=True)


def chunk_deltas(p0, R0, g0, tgt_pos, tgt_rotm, tgt_grip):
    """(30,7) relative actions for one chunk start. tgt_* are the next `steps`
    commanded targets (already sliced [t:t+steps]); padded to 30 by repeating last."""
    steps = len(tgt_pos)
    dpos = (R0.T @ (tgt_pos - p0).T).T                             # tool-frame translation
    daa = R.from_matrix(R0.T @ tgt_rotm).as_rotvec()               # axis-angle, tool frame
    dgrip = (tgt_grip - g0)[:, None]
    out = np.concatenate([dpos, daa, dgrip], axis=1).astype(np.float32)
    if steps < ACTION_LENGTH:
        out = np.concatenate([out, np.repeat(out[-1:], ACTION_LENGTH - steps, axis=0)])
    return out


def load(root: Path):
    root = Path(root)
    df = pq.read_table(root / "data/chunk-000/file-000.parquet").to_pandas()
    eps = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pandas()
    tasks = pq.read_table(root / "meta/tasks.parquet").to_pandas()
    info = json.loads((root / "meta/info.json").read_text())
    if info.get("fps") != 30:
        raise SystemExit(f"XR-1 action_length=30 assumes 30 fps; dataset is {info.get('fps')}")
    return df, eps, dict(zip(tasks.task_index, tasks.task)), info


def episode_arrays(df, ep_idx):
    rows = df[df.episode_index == ep_idx].sort_values("frame_index")
    eef = np.stack(rows["observation.eef_pose"].to_numpy()).astype(np.float64)
    st = np.stack(rows["observation.state"].to_numpy()).astype(np.float64)
    act = np.stack(rows["action"].to_numpy()).astype(np.float64)
    n = len(rows)
    pos = eef[:, :3] / 1000.0                                       # mm -> m
    rotm = rpy_deg_to_rotm(eef[:, 3:6])
    joints = np.deg2rad(st[:, :6])                                  # deg -> rad
    grip = st[:, 6]
    # commanded target for frame t := observed pose at t+1 (last repeats)
    nxt = np.minimum(np.arange(n) + 1, n - 1)
    task_index = int(rows["task_index"].iloc[0])
    return dict(n=n, pos=pos, rotm=rotm, joints=joints, grip=grip,
                a_pos=pos[nxt], a_rotm=rotm[nxt], a_grip=act[:, 6], task_index=task_index)


def episode_json(ep_idx, arr, instruction, root: Path):
    n = arr["n"]
    z1, z3, z6, z9 = np.zeros((n, 1)), np.zeros((n, 3)), np.zeros((n, 6)), np.tile(np.eye(3).ravel(), (n, 1))
    tolist = lambda a: np.asarray(a, dtype=np.float32).round(6).tolist()
    vid = lambda cam: str((root / "videos" / cam / "chunk-000" / f"file-{ep_idx:03d}.mp4").resolve())
    prompt = ("The following observations are captured from multiple views.\n"
              "# Ego View\n<image>\n# Left-Wrist View\n<image>\n"
              f"Generate robot actions for the task:\n{instruction}")
    return {
        "trajectory_type": "success",
        "time": f"fr5_episode_{ep_idx:03d}",
        "num_frames": n,
        "instruction": {"general": [{
            "images": ["observations.ego", "observations.wrist_left"],
            "conversations": [{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}],
        }]},
        "observations": {
            "ego":        [{"path": vid(CAMS["ego"]), "start": 0, "crop_bbox": None}],
            "wrist_left": [{"path": vid(CAMS["wrist_left"]), "start": 0, "crop_bbox": None}],
        },
        "proprios": {
            "left_ee_pos": tolist(arr["pos"]), "left_ee_rotm": tolist(arr["rotm"].reshape(n, 9)),
            "left_arm_joint": tolist(arr["joints"]), "left_gripper_pos": tolist(arr["grip"][:, None]),
            "right_ee_pos": tolist(z3), "right_ee_rotm": tolist(z9),
            "right_arm_joint": tolist(z6), "right_gripper_pos": tolist(z1), "waist_pos": tolist(z1),
        },
        "actions": {
            "left_ee_pos": tolist(arr["a_pos"]), "left_ee_rotm": tolist(arr["a_rotm"].reshape(n, 9)),
            "left_gripper_pos": tolist(arr["a_grip"][:, None]),
            "right_ee_pos": tolist(z3), "right_ee_rotm": tolist(z9), "right_gripper_pos": tolist(z1),
            "waist_pos": tolist(z1), "base_vel": tolist(z3),
        },
    }


class Stats:
    """Per-chunk-position mean/std of the packed relative actions (30,60) and
    q01/q99 of the packed state (1,60) — XR-1's normalisers, streamed per episode."""

    def __init__(self):
        self.n = 0
        self.s1 = np.zeros((ACTION_LENGTH, ACTION_DIM))
        self.s2 = np.zeros((ACTION_LENGTH, ACTION_DIM))
        self.states = []

    def add_episode(self, arr):
        n = arr["n"]
        for t in range(n):
            steps = min(ACTION_LENGTH, n - t)
            d = chunk_deltas(arr["pos"][t], arr["rotm"][t], arr["grip"][t],
                             arr["a_pos"][t:t + steps], arr["a_rotm"][t:t + steps], arr["a_grip"][t:t + steps])
            packed = np.zeros((ACTION_LENGTH, ACTION_DIM))
            packed[:, SL_POS], packed[:, SL_AA], packed[:, SL_GRIP] = d[:, 0:3], d[:, 3:6], d[:, 6:7]
            self.s1 += packed; self.s2 += packed * packed; self.n += 1
        st = np.zeros((n, STATE_DIM)); st[:, 0:6] = arr["joints"]; st[:, 7] = arr["grip"]
        self.states.append(st)

    def result(self):
        mean = self.s1 / max(self.n, 1)
        std = np.sqrt(np.maximum(self.s2 / max(self.n, 1) - mean * mean, 0.0))
        states = np.concatenate(self.states)
        q01, q99 = np.quantile(states, 0.01, axis=0)[None], np.quantile(states, 0.99, axis=0)[None]
        # a constant state dim (e.g. a joint that never moves) would give q01 == q99 != 0,
        # which validate_quantiles rejects; widen it by a hair so it is a valid range
        tie = (q99 <= q01) & ~((q01 == 0) & (q99 == 0))
        q99[tie] = q01[tie] + 1e-3
        return mean, std, q01, q99


def write_yaml(path: Path, json_dir: Path, mean, std, q01, q99, batch_size=48):
    def rows(a):
        return "\n".join("      - [" + ", ".join(f"{v:.6g}" for v in r) + "]" for r in np.asarray(a).tolist())
    text = f"""# @package _global_
# Generated by xr1_eef/fr5_to_xr1.py — FR5 pick-and-place, left-arm slot, 2 views.
data:
  type: BaseDataModule
  params:
    type: json
    max_steps: ${{trainer.max_steps}}
    train_datasets:
      batch_size: {batch_size}
      action_length: {ACTION_LENGTH}
      paths:
      - {json_dir.resolve()}
      mean:
{rows(mean)}
      std:
{rows(std)}
      q01:
{rows(q01)}
      q99:
{rows(q99)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="LeRobot-v2 dataset root (data/, meta/, videos/)")
    ap.add_argument("--out", required=True, help="output dir for json/ and configs/")
    ap.add_argument("--episodes", nargs="*", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="compute stats + schema, write nothing")
    args = ap.parse_args(argv)

    root, out = Path(args.root), Path(args.out)
    df, eps, tasks, info = load(root)
    ep_ids = args.episodes if args.episodes is not None else sorted(eps.episode_index.tolist())
    if not np.allclose(TOOL_FRAME_FIX, np.eye(3)):
        print(f"[tool frame] applying TOOL_FRAME_FIX =\n{TOOL_FRAME_FIX}")
    else:
        print("[tool frame] TOOL_FRAME_FIX = identity — XR-1's canonical EE-frame orientation "
              "is undocumented; verify before robot time (README §6 risk 1)")

    stats, n_frames = Stats(), 0
    (out / "json").mkdir(parents=True, exist_ok=True) if not args.dry_run else None
    for k, ep_idx in enumerate(ep_ids):
        arr = episode_arrays(df, ep_idx)
        instr = tasks[arr["task_index"]]
        stats.add_episode(arr); n_frames += arr["n"]
        if not args.dry_run:
            (out / "json" / f"episode_{ep_idx:03d}.json").write_text(
                json.dumps(episode_json(ep_idx, arr, instr, root)))
        if k < 3 or k == len(ep_ids) - 1:
            print(f"  ep {ep_idx:03d}: {arr['n']:5d} frames  {instr!r}")
    mean, std, q01, q99 = stats.result()
    print(f"\n{len(ep_ids)} episodes, {n_frames:,} frames -> {stats.n:,} chunk starts")
    print(f"  action std @ entry 0 : pos {std[0, 0:3].round(5)} m   rot {std[0, 3:6].round(5)} rad   grip {std[0, 6]:.3f}")
    print(f"  action std @ entry 29: pos {std[29, 0:3].round(4)} m   rot {std[29, 3:6].round(4)} rad")
    print(f"  state q01/q99 joints (rad): {q01[0, :6].round(3)} / {q99[0, :6].round(3)}   gripper {q01[0, 7]:.2f}/{q99[0, 7]:.2f}")
    if args.dry_run:
        print("dry run — nothing written")
        return
    write_yaml(out / "configs" / "fr5.yaml", out / "json", mean, std, q01, q99)
    print(f"wrote {out/'json'} ({len(ep_ids)} files) and {out/'configs'/'fr5.yaml'}")
    print("train with:  data=<copy fr5.yaml into xr1/configs/data/>  (see README §4 for the GPU note)")


if __name__ == "__main__":
    main()
