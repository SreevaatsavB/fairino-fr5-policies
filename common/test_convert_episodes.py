#!/usr/bin/env python3
"""test_convert_episodes.py — the two things that silently corrupt a conversion.

    python common/test_convert_episodes.py

1. episode_index must stay DENSE when _load_episode skips an episode. It used to
   come from enumerate(ep_dirs), so a skip left a hole: info.json said
   total_episodes=N while episode_index ran past N, and FR5Dataset.episode_split
   hands out range(total_episodes) — the highest real episodes matched no split
   and vanished from train AND val without a word.

2. max_sync_gap must drop camera frames whose nearest control row is stale (the
   blocking-gripper stall), and frame_index must still be the TRUE video frame
   position afterwards, because dataset._load_frame seeks the mp4 with it.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2

from convert_episodes import VIDEO_KEY, convert

FPS = 30.0
COLS = ([f"fr5_actual_j{i}" for i in range(1, 7)] + [f"fr5_cmd_j{i}" for i in range(1, 7)]
        + ["gripper_cmd", "fr5_eef_x_mm", "fr5_eef_y_mm", "fr5_eef_z_mm",
           "fr5_eef_rx_deg", "fr5_eef_ry_deg", "fr5_eef_rz_deg"])


def write_episode(root: Path, name: str, n_frames: int, stall_at=None,
                  stall_s=1.85, with_wrist_ts=True):
    """One synthetic raw episode. `stall_at` freezes the control log there while
    the camera keeps running — the blocking-gripper artefact."""
    d = root / name
    d.mkdir(parents=True)
    cam_ts = np.arange(n_frames) / FPS                      # clean 30 Hz video

    # control log: one row per camera frame, except the stall where it stops dead
    ctrl_ts = []
    t = 0.0
    for i in range(n_frames):
        if stall_at is not None and i == stall_at:
            t += stall_s                                    # loop blocked here
        ctrl_ts.append(t)
        t += 1.0 / FPS
    ctrl_ts = np.array(ctrl_ts)
    ctrl_ts = ctrl_ts[ctrl_ts <= cam_ts[-1] + 1e-9]         # log ends when video does

    n = len(ctrl_ts)
    df = pd.DataFrame({c: np.linspace(0, 1, n) for c in COLS})
    df["timestamp"] = ctrl_ts
    df.to_csv(d / "data.csv", index=False)

    np.save(d / "scene_cam_ts.npy", cam_ts)
    if with_wrist_ts:
        np.save(d / "wrist_cam_ts.npy", cam_ts)
    for cam in ("wrist_cam", "scene_cam"):
        vw = cv2.VideoWriter(str(d / f"{cam}.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), FPS, (32, 32))
        for i in range(n_frames):
            vw.write(np.full((32, 32, 3), i % 256, np.uint8))
        vw.release()
    (d / "meta.json").write_text(json.dumps({"instruction": "do the thing"}))


def read_out(out: Path):
    data = pq.read_table(out / "data/chunk-000/file-000.parquet").to_pandas()
    eps = pq.read_table(out / "meta/episodes/chunk-000/file-000.parquet").to_pandas()
    info = json.loads((out / "meta/info.json").read_text())
    return data, eps, info


def test_dense_episode_index(tmp: Path):
    """A skipped episode must not leave a hole in episode_index."""
    raw, out = tmp / "raw_dense", tmp / "out_dense"
    for i, wrist_ts in enumerate([True, False, True, True]):
        write_episode(raw, f"episode_{i:03d}", 40, with_wrist_ts=wrist_ts)
    convert(raw, out, False, None, "joint", cameras=(VIDEO_KEY,))

    data, eps, info = read_out(out)
    got = sorted(eps.episode_index.tolist())
    assert got == [0, 1, 2], f"episode_index not dense: {got}"
    assert info["total_episodes"] == 3, info["total_episodes"]
    # the split FR5Dataset would generate must cover every real episode
    assert set(got) <= set(range(info["total_episodes"])), (
        f"episode_split(range({info['total_episodes']})) misses {set(got) - set(range(info['total_episodes']))}")
    assert sorted(data.episode_index.unique().tolist()) == [0, 1, 2]
    # episode 1 was skipped, so ep 2 of the raw set is now index 1 — and its video
    # must have been copied under the NEW index, not the old one
    for i in range(3):
        assert (out / "videos" / VIDEO_KEY / "chunk-000" / f"file-{i:03d}.mp4").exists(), i
    print("  dense episode_index ....... ok  (3 episodes, 1 skipped, no holes)")


def test_sync_gap_drops_stall(tmp: Path):
    """Frames with no fresh control sample are dropped; frame_index stays true."""
    raw = tmp / "raw_sync"
    write_episode(raw, "episode_000", 120, stall_at=60)

    keep_all, _ = tmp / "out_keep", None
    convert(raw, keep_all, False, None, "joint", cameras=(VIDEO_KEY,))
    base, _, _ = read_out(keep_all)

    dropped_out = tmp / "out_drop"
    convert(raw, dropped_out, False, None, "joint", cameras=(VIDEO_KEY,),
            max_sync_gap=0.1)
    kept, _, _ = read_out(dropped_out)

    assert len(base) == 120, len(base)
    # nearest-neighbour keeps frames within the gap of the row on EITHER side of
    # the stall, so the dead span is the stall minus one gap at each end.
    n_stall = int(round((1.85 - 2 * 0.1) * FPS))
    assert len(kept) < len(base), "max_sync_gap dropped nothing"
    assert abs((len(base) - len(kept)) - n_stall) <= 2, (
        f"expected ~{n_stall} frames dropped, got {len(base) - len(kept)}")

    # default must stay exactly the historical behaviour
    assert base.frame_index.tolist() == list(range(120)), "default path changed"

    # frame_index must remain the TRUE video position (dataset seeks with it),
    # so it is strictly increasing with a hole where the stall was — NOT renumbered
    fi = kept.frame_index.to_numpy()
    assert (np.diff(fi) > 0).all(), "frame_index not increasing"
    assert fi.max() <= 119, f"frame_index {fi.max()} past end of video"
    assert np.diff(fi).max() > 1, "no hole left by the drop — did it renumber?"
    assert fi[0] == 0
    print(f"  max_sync_gap stall drop ... ok  (120 -> {len(kept)} frames, "
          f"hole of {np.diff(fi).max()} at the stall, frame_index true)")


def test_extract_frames_follows_holes(tmp: Path):
    """--extract-frames must write the JPEGs the dataset will actually ask for."""
    raw, out = tmp / "raw_ex", tmp / "out_ex"
    write_episode(raw, "episode_000", 90, stall_at=45)
    convert(raw, out, True, None, "joint", cameras=(VIDEO_KEY,), max_sync_gap=0.1)
    data, _, _ = read_out(out)
    jpg_dir = out / "frames" / VIDEO_KEY / "ep-000"
    on_disk = {int(p.stem) for p in jpg_dir.glob("*.jpg")}
    wanted = set(data.frame_index.tolist())
    assert wanted <= on_disk, f"dataset would miss {sorted(wanted - on_disk)[:5]}"
    print(f"  extract-frames alignment .. ok  ({len(on_disk)} jpgs cover all "
          f"{len(wanted)} rows)")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="convtest_"))
    try:
        test_dense_episode_index(tmp)
        test_sync_gap_drops_stall(tmp)
        test_extract_frames_follows_holes(tmp)
        print("\nall convert_episodes checks passed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
