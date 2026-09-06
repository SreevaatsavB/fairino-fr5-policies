"""
convert_episodes.py — raw teleop episodes  →  LeRobot v3.0 dataset.

The recorder writes one folder per episode:

    episodes/episode_000/
        data.csv            ~100 Hz log: so101 leader, fr5 cmd/actual joints,
                            gripper_norm, velocities, eef pose
        wrist_cam.mp4       ~30 Hz wrist camera
        wrist_cam_ts.npy    one timestamp per video frame
        scene_cam.*         (unused by the policy)
        meta.json           language instruction, camera intrinsics, ...

The ACT pipeline (training/dataset.py) consumes a LeRobot v3.0 dataset:

    <out>/meta/info.json
    <out>/meta/tasks.parquet
    <out>/meta/episodes/chunk-000/file-000.parquet
    <out>/data/chunk-000/file-000.parquet
    <out>/videos/observation.images.wrist_cam/chunk-000/file-<ep>.mp4

The CSV runs at ~100 Hz but the policy is trained/run at the 30 Hz camera
rate, so we resample the CSV onto the camera timestamps: exactly one dataset
row per video frame (nearest-in-time CSV sample). This keeps `frame_index`
aligned with the video and with the optional pre-extracted frames.

Usage:
    python tools/convert_episodes.py \
        --episodes episodes \
        --out ../so101-fr5-teleop/lerobot_dataset \
        --extract-frames

    # quick pipeline smoke test (few frames per episode, self-contained out dir)
    python tools/convert_episodes.py --out _smoke_dataset --extract-frames --max-frames 40
"""

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import os
import numpy as np
import pandas as pd

# FIX-1 SYNC: shift camera timestamps by this many seconds before pairing each frame
# to a control row. Measured cam-content lag was +1 frame (+33 ms), so pair each image
# with control ~33 ms EARLIER -> a NEGATIVE shift. Set via FR5_SYNC_SHIFT_MS (e.g. -33).
_SYNC_SHIFT_S = float(os.environ.get("FR5_SYNC_SHIFT_MS", "0")) / 1000.0
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import cv2
except ImportError:  # only needed for --extract-frames
    cv2 = None


CODEBASE_VERSION = "v3.0"
FPS = 30  # policy control rate; matches the camera, not the CSV

# CSV column -> dataset feature mapping
STATE_COLS = [f"fr5_actual_j{i}" for i in range(1, 7)] + ["gripper_norm"]  # observation.state (7)
CMD_COLS = [f"fr5_cmd_j{i}" for i in range(1, 7)]               # action joints (6)
GRIPPER_COL = "gripper_norm"                                    # action[6]
EEF_COLS = ["fr5_eef_x_mm", "fr5_eef_y_mm", "fr5_eef_z_mm",
            "fr5_eef_rx_deg", "fr5_eef_ry_deg", "fr5_eef_rz_deg"]  # observation.eef_pose (6)

# action feature names per action space (action_dim stays 7 either way)
ACTION_NAMES = {
    "joint":     CMD_COLS + [GRIPPER_COL],
    "delta_eef": ["delta_eef_x_mm", "delta_eef_y_mm", "delta_eef_z_mm",
                  "delta_eef_rx_deg", "delta_eef_ry_deg", "delta_eef_rz_deg",
                  GRIPPER_COL],
    "abs_eef":   EEF_COLS + [GRIPPER_COL],   # absolute EEF pose trajectory + gripper
}
# observation.state names per action space: joint-space for joint/delta_eef,
# EEF-space (the same 6-D pose) for abs_eef so the policy reasons in task space.
STATE_NAMES = {
    "joint":     STATE_COLS,
    "delta_eef": STATE_COLS,
    "abs_eef":   EEF_COLS + [GRIPPER_COL],
}

VIDEO_KEY = "observation.images.wrist_cam"
SCENE_KEY = "observation.images.scene_cam"
# camera key -> (source mp4 filename, source timestamp filename)
CAMERAS = {
    VIDEO_KEY: ("wrist_cam.mp4", "wrist_cam_ts.npy"),
    SCENE_KEY: ("scene_cam.mp4", "scene_cam_ts.npy"),
}


def _nearest_indices(csv_ts: np.ndarray, cam_ts: np.ndarray) -> np.ndarray:
    """For each camera timestamp, the index of the nearest CSV row."""
    pos = np.searchsorted(csv_ts, cam_ts)
    pos = np.clip(pos, 1, len(csv_ts) - 1)
    left, right = csv_ts[pos - 1], csv_ts[pos]
    choose_left = (cam_ts - left) <= (right - cam_ts)
    return pos - choose_left.astype(int)


def _load_episode(ep_dir: Path, max_frames: int | None, action_space: str = "joint",
                  task_override: str | None = None, cameras=(VIDEO_KEY,),
                  max_sync_gap: float | None = None):
    """Resample one raw episode onto its camera timestamps.

    max_sync_gap:
      Drop camera frames whose nearest control row is more than this many seconds
      away. None (default) keeps every frame — the historical behaviour.
      Set it when the recorder's control loop stalls: a blocking gripper call
      freezes the CSV for ~1.85 s while the cameras keep running at 30 Hz, so ~55
      frames all resample to the SAME control row and become identical zero-motion
      targets at exactly the visual moment of the grasp. Train on those and the
      policy learns to freeze when it sees a grasp about to happen — and since it
      then never moves, the scene never changes and it cannot recover.

    action_space:
      "joint"     — action = absolute commanded joint angles + gripper (default).
      "delta_eef" — action = Cartesian TCP pose delta (eef[t+1]-eef[t]) + gripper.
                    Generalizes far better but needs IK / ServoCart at deploy
                    (see docs/action_spaces.md). action_dim is 7 either way.

    Returns (frame_dict, task_str) or None if the episode can't be used.
    """
    csv_path = ep_dir / "data.csv"
    ts_path = ep_dir / "wrist_cam_ts.npy"
    if not csv_path.exists() or not ts_path.exists():
        print(f"  [skip] {ep_dir.name}: missing data.csv or wrist_cam_ts.npy")
        return None

    df = pd.read_csv(csv_path)
    cam_ts = np.load(ts_path).astype(np.float64)

    # schema alias: the derived single-marker episodes carry the gripper command
    # as `gripper_cmd` (already normalised to [0, 1]); the converter expects
    # `gripper_norm`. Map it across so STATE_COLS / GRIPPER_COL resolve.
    if "gripper_norm" not in df.columns and "gripper_cmd" in df.columns:
        df["gripper_norm"] = df["gripper_cmd"]

    # actual-joint / eef columns have a couple of NaN rows at startup — fill them
    # so nearest-neighbour resampling never lands on a NaN.
    fill_cols = STATE_COLS + EEF_COLS
    df[fill_cols] = df[fill_cols].ffill().bfill()

    csv_ts = df["timestamp"].to_numpy(np.float64)
    shifted = cam_ts + _SYNC_SHIFT_S
    idx = _nearest_indices(csv_ts, shifted)                  # FIX-1: apply sync shift
    # which VIDEO frame each row comes from — no longer implicit once max_sync_gap
    # drops frames, and frame_index must stay the true video position because
    # dataset._load_frame seeks with it (cap.set(POS_FRAMES) / {idx:06d}.jpg).
    keep = np.arange(len(idx), dtype=np.int64)

    if max_frames is not None:
        idx, keep = idx[:max_frames], keep[:max_frames]

    # Cap to the SHORTEST used camera so every dataset row has a real frame in
    # EVERY view (the two cams can differ by a frame; without this the last row
    # would reference a scene frame that doesn't exist).
    for cam_key in cameras:
        _, tsf = CAMERAS[cam_key]
        tsf_path = ep_dir / tsf
        if tsf_path.exists():
            n_cam = len(np.load(tsf_path))
            idx, keep = idx[:n_cam], keep[:n_cam]

    if max_sync_gap is not None:
        fresh = np.abs(shifted[keep] - csv_ts[idx]) <= max_sync_gap
        dropped = int((~fresh).sum())
        if dropped:
            print(f"  [sync] {ep_dir.name}: dropped {dropped}/{len(fresh)} frames "
                  f"with no control sample within {max_sync_gap*1000:.0f} ms")
        idx, keep = idx[fresh], keep[fresh]
        if len(idx) == 0:
            print(f"  [skip] {ep_dir.name}: every frame failed the sync gap")
            return None

    rows = df.iloc[idx].reset_index(drop=True)

    state = rows[STATE_COLS].to_numpy(np.float32)
    eef = rows[EEF_COLS].to_numpy(np.float32)
    gripper = rows[[GRIPPER_COL]].to_numpy(np.float32)

    if action_space == "delta_eef":
        # action[t] = eef[t+1] - eef[t]  (TCP pose delta within the episode);
        # the final frame has no t+1, so its delta is zero. Orientation deltas
        # (cols 3:6) are wrapped to [-180, 180] so a +179°/-181° pair (the same
        # physical rotation) doesn't average to nonsense under temporal ensembling.
        delta = np.zeros_like(eef)
        delta[:-1] = eef[1:] - eef[:-1]
        delta[:, 3:6] = (delta[:, 3:6] + 180.0) % 360.0 - 180.0
        action = np.concatenate([delta, gripper], axis=1)
        obs_state = state                                       # joint-space state
    elif action_space == "abs_eef":
        # action[t] = absolute EEF pose at t + gripper (the task-space trajectory);
        # state is ALSO the absolute EEF pose so the policy reasons in task space.
        # Deploy IK-solves each predicted pose. Avoids the delta mean-collapse.
        action = np.concatenate([eef, gripper], axis=1)
        obs_state = np.concatenate([eef, gripper], axis=1)
    else:  # "joint" (default)
        action = np.concatenate([rows[CMD_COLS].to_numpy(np.float32), gripper], axis=1)
        obs_state = state                                       # joint-space state

    frame = {
        "observation.state": list(obs_state),
        "action": list(action),
        "observation.eef_pose": list(eef),
        # the TRUE video frame each row came from — identity unless max_sync_gap
        # dropped frames. dataset._load_frame seeks the video with this value.
        "frame_index": keep,
        "timestamp": (cam_ts[keep] - cam_ts[keep[0]]).astype(np.float32),
        "video_frame": keep,
    }

    meta_path = ep_dir / "meta.json"
    task = "pick up the block and place it in the bin"
    if meta_path.exists():
        m = json.loads(meta_path.read_text())
        task = m.get("language_instruction") or m.get("instruction") or task
    # Per-episode instructions from meta.json flow through UNCHANGED — each unique
    # string gets its own task_index, which is what makes the dataset genuinely
    # language-conditioned (the 2026-07 raw set carries 400 unique phrasings over
    # 9 canonical tasks). --task collapses everything to ONE string; that is only
    # for single-task smoke tests and destroys the language signal — never use it
    # on the multi-task data.
    if task_override is not None:
        task = task_override

    return frame, task


def _extract_frames(video: Path, frame_indices, out_dir: Path, stride: int = 1):
    """Dump frames of `video` as JPEGs named by their TRUE frame index
    (000000.jpg, then every `stride`-th: 000005.jpg, ...). Frames are read
    sequentially (fast) but only every `stride`-th is written, so the dataset's
    matching frame_stride loads exactly these without wasted disk.

    `frame_indices` are the video positions the dataset will actually ask for
    (rows' frame_index). Striding over that list rather than over range(n) keeps
    the JPEGs aligned when max_sync_gap has punched holes in it."""
    if cv2 is None:
        raise RuntimeError("--extract-frames needs opencv-python (cv2) installed")
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = {int(f) for f in np.asarray(frame_indices)[::stride]}
    cap = cv2.VideoCapture(str(video))
    written = 0
    for i in range(max(wanted) + 1 if wanted else 0):
        ret, frame = cap.read()
        if not ret:
            break
        if i in wanted:
            cv2.imwrite(str(out_dir / f"{i:06d}.jpg"), frame)
            written += 1
    cap.release()
    return written


def convert(episodes_dir: Path, out_dir: Path, extract_frames: bool,
            max_frames: int | None, action_space: str = "joint",
            glob_pattern: str = "episode_*", task_override: str | None = None,
            cameras=(VIDEO_KEY,), extract_stride: int = 1, exclude=(),
            max_sync_gap: float | None = None):
    if action_space not in ACTION_NAMES:
        raise SystemExit(f"unknown action_space {action_space!r}; "
                         f"use one of {list(ACTION_NAMES)}")
    ep_dirs = sorted(p for p in episodes_dir.glob(glob_pattern) if p.is_dir())
    if exclude:
        excl = set(exclude); ep_dirs = [p for p in ep_dirs if p.name not in excl]
        print(f"excluded {len(excl)} episode(s): {sorted(excl)}")
    if not ep_dirs:
        raise SystemExit(f"no {glob_pattern!r} folders under {episodes_dir}")

    print(f"found {len(ep_dirs)} episode(s) under {episodes_dir}  "
          f"(action_space={action_space}, cameras={list(cameras)})")

    data_dir = out_dir / "data" / "chunk-000"
    ep_meta_dir = out_dir / "meta" / "episodes" / "chunk-000"
    video_dirs = {k: out_dir / "videos" / k / "chunk-000" for k in cameras}
    frames_roots = {k: out_dir / "frames" / k for k in cameras}
    for d in (data_dir, ep_meta_dir, *video_dirs.values()):
        d.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    ep_records: list[dict] = []
    tasks: dict[str, int] = {}
    extract_jobs: list[tuple] = []   # (video, frame_ids, out_dir, stride) — parallel below
    cursor = 0  # running absolute row index across episodes
    # episode_index MUST stay dense. It used to be enumerate(ep_dirs), so every
    # episode _load_episode skipped left a hole: info.json reported
    # total_episodes=246 while episode_index ran 0..264, and FR5Dataset.episode_split
    # hands out range(total_episodes) — so the 19 highest real episodes matched no
    # split and were silently dropped from BOTH train and val. Only advance on a
    # successful load. (--exclude is safe either way: it filters ep_dirs up front.)
    ep_idx = -1

    for ep_dir in ep_dirs:
        loaded = _load_episode(ep_dir, max_frames, action_space, task_override,
                               cameras=cameras, max_sync_gap=max_sync_gap)
        if loaded is None:
            continue
        ep_idx += 1
        frame, task = loaded
        n = len(frame["frame_index"])
        task_index = tasks.setdefault(task, len(tasks))

        for i in range(n):
            all_rows.append({
                "observation.state": frame["observation.state"][i],
                "action": frame["action"][i],
                "observation.eef_pose": frame["observation.eef_pose"][i],
                "timestamp": float(frame["timestamp"][i]),
                "frame_index": int(frame["frame_index"][i]),
                "episode_index": ep_idx,
                "index": cursor + i,
                "task_index": task_index,
            })

        # copy each camera's video so dataset.py can read frames, and QUEUE the
        # frame extraction (all jobs run in parallel after the loop — see below).
        for cam_key in cameras:
            src_name, _ = CAMERAS[cam_key]
            src_video = ep_dir / src_name
            dst_video = video_dirs[cam_key] / f"file-{ep_idx:03d}.mp4"
            if src_video.exists():
                shutil.copyfile(src_video, dst_video)
                if extract_frames:
                    extract_jobs.append((dst_video, frame["frame_index"],
                                         frames_roots[cam_key] / f"ep-{ep_idx:03d}",
                                         extract_stride))
            else:
                print(f"  [warn] {ep_dir.name}: no {src_name}")

        ep_records.append({
            "episode_index": ep_idx,
            "dataset_from_index": cursor,
            "dataset_to_index": cursor + n,
            "length": n,
            "tasks": [task],
            "task": task,
        })
        cursor += n
        print(f"  {ep_dir.name} -> ep {ep_idx}: {n} frames "
              f"[{cursor - n}:{cursor})  task={task!r}")

    if not all_rows:
        raise SystemExit("no frames produced — nothing written")

    # ---- parallel frame extraction (the slow part) ----
    # OpenCV releases the GIL during decode/encode, so a thread pool genuinely
    # runs on all cores. Full quality (identical cv2.imwrite), just parallel.
    if extract_frames and extract_jobs:
        workers = min(len(extract_jobs), max(4, (os.cpu_count() or 4)))
        n_jobs = len(extract_jobs)
        print(f"extracting frames from {n_jobs} videos with {workers} threads "
              f"(full quality, parallel)...")
        done = 0
        def _job(a):
            video, frame_ids, out_dir, stride = a
            return _extract_frames(video, frame_ids, out_dir, stride)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            total_extracted = 0
            for got in ex.map(_job, extract_jobs):
                total_extracted += got
                done += 1
                if done % 25 == 0 or done == n_jobs:
                    print(f"    {done}/{n_jobs} videos  ({total_extracted} frames)", flush=True)

    total_frames = cursor
    total_episodes = len(ep_records)

    # ---- data/chunk-000/file-000.parquet ----
    data_df = pd.DataFrame(all_rows)
    pq.write_table(pa.Table.from_pandas(data_df, preserve_index=False),
                   data_dir / "file-000.parquet")

    # ---- meta/episodes/chunk-000/file-000.parquet ----
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(ep_records), preserve_index=False),
                   ep_meta_dir / "file-000.parquet")

    # ---- meta/tasks.parquet ----
    tasks_df = pd.DataFrame(
        {"task": list(tasks.keys()), "task_index": list(tasks.values())}
    ).set_index("task_index")
    pq.write_table(pa.Table.from_pandas(tasks_df.reset_index(), preserve_index=False),
                   out_dir / "meta" / "tasks.parquet")

    # ---- meta/info.json ----
    action_names = ACTION_NAMES[action_space]
    state_names  = STATE_NAMES[action_space]
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": "fairino_fr5",
        "fps": FPS,
        "action_space": action_space,   # "joint" | "delta_eef" — deploy reads this to pick execution
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_chunks": 1,
        "chunks_size": 1000,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [len(state_names)],
                                   "names": state_names},
            "action": {"dtype": "float32", "shape": [len(action_names)],
                       "names": action_names},
            "observation.eef_pose": {"dtype": "float32", "shape": [len(EEF_COLS)],
                                     "names": EEF_COLS},
            **{k: {"dtype": "video", "shape": [480, 640, 3]} for k in cameras},
        },
        "camera_names": [k.split(".")[-1] for k in cameras],
    }
    (out_dir / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    print(f"\nwrote LeRobot dataset to {out_dir}")
    print(f"  episodes={total_episodes}  frames={total_frames}  "
          f"tasks={len(tasks)}  fps={FPS}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", default="episodes",
                    help="folder containing episode_* directories")
    ap.add_argument("--out", default="../so101-fr5-teleop/lerobot_dataset",
                    help="output LeRobot dataset root")
    ap.add_argument("--extract-frames", action="store_true",
                    help="pre-extract aligned JPEG frames for fast training I/O")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap frames per episode (for quick pipeline tests)")
    ap.add_argument("--action-space", choices=["joint", "delta_eef", "abs_eef"], default="joint",
                    help="joint: absolute joint-angle commands (default, no IK at deploy). "
                         "delta_eef: Cartesian TCP pose deltas (better generalization; "
                         "needs IK/ServoCart at deploy — see docs/action_spaces.md)")
    ap.add_argument("--exclude", default="",
                    help="comma list of episode basenames to DROP (FIX-2), "
                         "e.g. episode_064,episode_106")
    ap.add_argument("--glob", default="episode_*",
                    help="folder glob for episodes (default 'episode_*'; "
                         "use 'ep*' for the derived single-marker dataset)")
    ap.add_argument("--task", default=None,
                    help="override the language instruction for ALL episodes — "
                         "collapses per-colour labels to one task_index")
    ap.add_argument("--cameras", default="wrist",
                    help="comma list of cameras: 'wrist', 'scene', or 'wrist,scene'")
    ap.add_argument("--extract-stride", type=int, default=1,
                    help="extract only every Kth frame (matches dataset.frame_stride) "
                         "— the 30 Hz capture is heavily oversampled for ACT")
    ap.add_argument("--max-sync-gap", type=float, default=None,
                    help="seconds; drop camera frames with no control sample that "
                         "close. Unset = keep everything. Use ~0.1 on recordings "
                         "where a blocking gripper call stalls the control loop "
                         "(the cameras keep rolling, so ~55 frames collapse onto "
                         "one frozen control row and teach the policy to freeze)")
    args = ap.parse_args()

    cam_map = {"wrist": VIDEO_KEY, "scene": SCENE_KEY}
    cams = tuple(cam_map[c.strip()] for c in args.cameras.split(",") if c.strip())

    excl = tuple(s.strip() for s in (args.exclude or "").split(",") if s.strip())
    convert(Path(args.episodes), Path(args.out),
            args.extract_frames, args.max_frames, args.action_space,
            glob_pattern=args.glob, task_override=args.task, cameras=cams,
            extract_stride=args.extract_stride, exclude=excl,
            max_sync_gap=args.max_sync_gap)


if __name__ == "__main__":
    main()
