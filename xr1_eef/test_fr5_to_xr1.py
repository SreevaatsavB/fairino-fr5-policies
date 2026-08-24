"""
test_fr5_to_xr1.py — self-check for the FR5 -> XR-1 conversion.

    python xr1_eef/test_fr5_to_xr1.py

No framework. Uses a synthetic LeRobot-v2 dataset for the schema/stat checks and,
when /tmp/ds_v2_edit (the real 400-episode parquet) is present, re-runs the
joint-6 Euler-convention test on real data.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fr5_to_xr1 as X  # noqa: E402

REAL = Path("/tmp/ds_v2_edit")


def test_euler_convention_is_extrinsic_xyz():
    """Fairino robot_types.h: rx/ry/rz rotate about the FIXED X, Y, Z axes."""
    def Rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    def Ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    def Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    rng = np.random.default_rng(0)
    for rpy in rng.uniform(-180, 180, size=(50, 3)):
        r = np.deg2rad(rpy)
        want = Rz(r[2]) @ Ry(r[1]) @ Rx(r[0])
        assert np.allclose(X.rpy_deg_to_rotm(rpy[None])[0], want, atol=1e-9)
    print("  rpy_deg_to_rotm == Rz(rz)·Ry(ry)·Rx(rx) on 50 random poses")


def test_wrap_around_is_harmless():
    """Our data has 41 jumps of rx across ±180 deg; the matrix must not notice."""
    a = X.rpy_deg_to_rotm(np.array([[179.95, 2.0, 1.5]]))[0]
    b = X.rpy_deg_to_rotm(np.array([[-179.95, 2.0, 1.5]]))[0]
    ang = np.rad2deg(np.linalg.norm(R.from_matrix(a.T @ b).as_rotvec()))
    assert ang < 0.2, f"±180 wrap produced a {ang:.2f} deg jump"
    # and the inverse used at deploy round-trips
    rpy = np.array([[-132.83, 2.06, 1.67], [170.0, -40.0, 80.0]])
    back = X.rotm_to_rpy_deg(X.rpy_deg_to_rotm(rpy))
    assert np.allclose(back, rpy, atol=1e-6), back
    print("  ±180 wrap: 0.1 deg apart, and rotm -> rpy round-trips exactly")


def test_deltas_match_xr1_recover():
    """chunk_deltas must be the exact inverse of XR-1's io.recover_action:
         pos_target  = pos0 + Δpos @ rotm0.T
         rotm_target = rotm0 @ aa2rotm(Δaa)
    i.e. what the robot PC will do with the model output."""
    rng = np.random.default_rng(1)
    p0 = rng.normal(size=3); R0 = R.random(random_state=2).as_matrix(); g0 = 0.0
    tgt_p = p0 + rng.normal(scale=0.02, size=(5, 3))
    tgt_R = np.stack([(R.from_rotvec(rng.normal(scale=0.1, size=3)) * R.from_matrix(R0)).as_matrix() for _ in range(5)])
    tgt_g = np.array([0, 0, 1, 1, 1], dtype=float)
    d = X.chunk_deltas(p0, R0, g0, tgt_p, tgt_R, tgt_g)
    assert d.shape == (30, 7)
    for i in range(5):
        p_rec = p0 + d[i, 0:3] @ R0.T                       # recover_action, verbatim
        R_rec = R0 @ R.from_rotvec(d[i, 3:6]).as_matrix()
        assert np.allclose(p_rec, tgt_p[i], atol=1e-5), i
        assert np.allclose(R_rec, tgt_R[i], atol=1e-5), i
        assert np.isclose(g0 + d[i, 6], tgt_g[i])
    assert np.allclose(d[5:], d[4]), "padding must repeat the last real entry"
    print("  chunk_deltas <-> recover_action round-trip exact; padding repeats last")


def _synthetic_lerobot(root: Path, n_eps=3, n_frames=40):
    """Minimal LeRobot-v2 layout the converter reads: data parquet, episodes, tasks, info."""
    rng = np.random.default_rng(0)
    rows, eps = [], []
    idx = 0
    for e in range(n_eps):
        base = np.array([0, 300, 380, -130, 2, 1], dtype=float)
        for f in range(n_frames):
            eef = base + np.array([f * 0.5, f * 0.3, -f * 0.2, f * 0.1, 0, f * 0.05])
            st = np.concatenate([rng.uniform(-100, 100, 6), [float(f > n_frames // 2)]])
            act = np.concatenate([st[:6] + 0.1, [float(f > n_frames // 2 - 2)]])
            rows.append({"observation.state": st.astype(np.float32), "observation.eef_pose": eef.astype(np.float32),
                         "action": act.astype(np.float32), "episode_index": e, "frame_index": f,
                         "index": idx, "task_index": e % 2, "timestamp": f / 30})
            idx += 1
        eps.append({"episode_index": e, "dataset_from_index": idx - n_frames, "dataset_to_index": idx,
                    "length": n_frames, "task": "stale per-episode string", "tasks": ["x"]})
    (root / "data/chunk-000").mkdir(parents=True); (root / "meta/episodes/chunk-000").mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(root / "data/chunk-000/file-000.parquet", index=False)
    pd.DataFrame(eps).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet", index=False)
    pd.DataFrame({"task_index": [0, 1], "task": ["Pick up each blue block and put it in the brown tray.",
                                                 "Pick up each cream block and put it in the wooden tray."]}
                 ).to_parquet(root / "meta/tasks.parquet", index=False)
    (root / "meta/info.json").write_text(json.dumps({"fps": 30, "total_episodes": n_eps}))


def test_converter_schema_and_stats():
    with tempfile.TemporaryDirectory() as td:
        root, out = Path(td) / "ds", Path(td) / "out"
        _synthetic_lerobot(root)
        X.main([str(root), "--out", str(out)])
        files = sorted((out / "json").glob("*.json"))
        assert len(files) == 3, files
        d = json.loads(files[0].read_text())
        # schema (docs/data_format.md): required keys, one entry per frame
        for k in ("trajectory_type", "num_frames", "instruction", "observations", "proprios", "actions"):
            assert k in d, k
        n = d["num_frames"]
        for k in ("left_ee_pos", "left_ee_rotm", "left_arm_joint", "left_gripper_pos",
                  "right_ee_pos", "right_ee_rotm", "right_arm_joint", "right_gripper_pos", "waist_pos"):
            assert len(d["proprios"][k]) == n, k
        for k in ("left_ee_pos", "left_ee_rotm", "left_gripper_pos", "right_ee_pos",
                  "right_ee_rotm", "right_gripper_pos", "waist_pos", "base_vel"):
            assert len(d["actions"][k]) == n, k
        assert len(d["proprios"]["left_ee_rotm"][0]) == 9 and len(d["proprios"]["left_arm_joint"][0]) == 6
        # units
        assert max(abs(v) for row in d["proprios"]["left_ee_pos"] for v in row) < 5, "ee_pos must be metres"
        assert max(abs(v) for row in d["proprios"]["left_arm_joint"] for v in row) < 2 * np.pi, "joints must be radians"
        # instruction comes from task_index -> tasks.parquet, NOT the stale episodes.task column
        prompt = d["instruction"]["general"][0]["conversations"][0]["value"]
        assert "Pick up each blue block" in prompt and "stale" not in prompt
        assert prompt.count("<image>") == 2 == len(d["instruction"]["general"][0]["images"])
        # action[t] is the next observed pose; last repeats
        assert d["actions"]["left_ee_pos"][0] == d["proprios"]["left_ee_pos"][1]
        assert d["actions"]["left_ee_pos"][-1] == d["proprios"]["left_ee_pos"][-1]
        # stats file: shapes XR-1's validate_stats / validate_quantiles demand
        y = (out / "configs" / "fr5.yaml").read_text()
        blocks, cur = {}, None
        for line in y.splitlines():                       # rows are "      - [ ... ]" under "      key:"
            if line.strip() in ("mean:", "std:", "q01:", "q99:"):
                cur = line.strip()[:-1]; blocks[cur] = []
            elif cur and line.strip().startswith("- ["):
                blocks[cur].append(line.strip()[3:-1])
            elif cur and line.strip() and not line.strip().startswith("- ["):
                cur = None
        assert len(blocks["mean"]) == 30 and len(blocks["std"]) == 30, "mean/std must be (30, 60)"
        assert len(blocks["q01"]) == 1 and len(blocks["q99"]) == 1, "q01/q99 must be (1, 60)"
        assert all(len(b.split(",")) == 60 for k in blocks for b in blocks[k]), "every row must have 60 dims"
        q01 = np.array([float(v) for v in blocks["q01"][0].split(",")]); q99 = np.array([float(v) for v in blocks["q99"][0].split(",")])
        pad = (q01 == 0) & (q99 == 0)
        assert (q99[~pad] > q01[~pad]).all(), "validate_quantiles: q99 > q01 outside padding"
        assert pad[8:16].all() and pad[16:].all(), "right arm / waist state dims must be zero padding"
        print("  3 episodes -> XR-1 JSON schema OK, units OK, 2-view prompt, stats (30,60)/(1,60) valid")


def test_real_data_joint6_roll_if_present():
    """On frames where ONLY joint 6 moved, the relative rotation in the tool frame
    must be a pure roll about tool z. Decides the Euler convention empirically."""
    if not (REAL / "data/chunk-000/file-000.parquet").exists():
        print("  (skipped: real parquet not at /tmp/ds_v2_edit)"); return
    import pyarrow.parquet as pq
    df = pq.read_table(REAL / "data/chunk-000/file-000.parquet").to_pandas()
    e = np.stack(df["observation.eef_pose"].to_numpy()).astype(np.float64)
    s = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    ep = df["episode_index"].to_numpy()
    dj = np.abs(np.diff(s[:, :6], axis=0)); same = ep[1:] == ep[:-1]
    sel = np.where(same & (dj[:, 5] > 0.4) & (dj[:, :5].max(1) < 0.05))[0][:300]
    assert len(sel) > 50, "not enough J6-only frames"
    Rm = X.rpy_deg_to_rotm(e[:, 3:6])
    z = []
    for i in sel:
        v = R.from_matrix(Rm[i].T @ Rm[i + 1]).as_rotvec()
        z.append(abs(v[2]) / (np.linalg.norm(v) + 1e-12))
    assert np.mean(z) > 0.99, f"|axis.z| = {np.mean(z):.3f}: Euler convention is wrong"
    print(f"  real data: J6-only motion is a pure tool-z roll, |axis·z| = {np.mean(z):.3f} on {len(sel)} frames")


if __name__ == "__main__":
    for fn in (test_euler_convention_is_extrinsic_xyz, test_wrap_around_is_harmless,
               test_deltas_match_xr1_recover, test_converter_schema_and_stats,
               test_real_data_joint6_roll_if_present):
        print(f"{fn.__name__}:")
        fn()
    print("\nall fr5->xr1 checks passed")
