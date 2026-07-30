"""
test_rebalance_instructions.py — self-check for the instruction collapse.

    python tools/test_rebalance_instructions.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rebalance_instructions import NATURAL  # noqa: E402

TOOL = REPO / "tools" / "rebalance_instructions.py"
FRAMES = 3


def make_dataset(root: Path, n_per=4):
    """Every episode a unique phrasing — the 1:1 binding the real set has."""
    canonical, rows, tasks = {}, [], []
    ep = 0
    for canon in NATURAL:
        for _ in range(n_per):
            canonical[ep] = {"canonical": canon,
                             "instruction": f"unique phrasing {ep} for {canon}"}
            tasks.append(canonical[ep]["instruction"])
            rows += [{"episode_index": ep, "frame_index": f, "task_index": ep,
                      "index": len(rows) + f} for f in range(FRAMES)]
            ep += 1
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "canonical_tasks.json").write_text(json.dumps(canonical))
    pd.DataFrame({"task_index": range(len(tasks)), "task": tasks}).to_parquet(
        root / "meta" / "tasks.parquet", index=False)
    pd.DataFrame(rows).to_parquet(root / "data" / "chunk-000" / "file-000.parquet",
                                  index=False)
    return canonical


def test_natural_map():
    """9 canonical tasks -> 9 distinct, grammatical sentences, colour + tray only."""
    assert len(NATURAL) == 9, len(NATURAL)
    assert len(set(NATURAL.values())) == 9, "rewrites must be distinct"
    for canon, nat in NATURAL.items():
        assert nat == nat.strip() and "_" not in nat and "  " not in nat, nat
        assert 5 <= len(nat.split()) <= 14, f"{nat!r} is {len(nat.split())} words"
        # grammar: sentence case and a full stop
        assert nat[0].isupper(), f"{nat!r} does not start with a capital"
        assert nat.endswith("."), f"{nat!r} has no full stop"
        # number agreement: distributive singular, so it is correct for 1 object or
        # 4. A bare plural would be ungrammatical for the 146 single-object episodes.
        assert " each " in nat and " it " in nat, f"{nat!r} is not distributive"
        assert " them " not in nat and " blocks " not in nat, \
            f"{nat!r} uses a plural — wrong for the 146 single-object episodes"
        for colour in ("blue", "brown", "cream"):          # colour must survive
            if colour in canon.split("objects")[0].lower():
                assert colour in nat, (canon, nat)
        for leak in ("single", "both", "two", "three", "cuboid"):
            assert leak not in nat, f"{leak!r} leaked into {nat!r} — camera's job"
    print("  9 sentences: sentence case, full stop, distributive singular,")
    print("  5-14 words, no counts/shapes leaked")


def test_collapse():
    """A 1:1 vocabulary collapses to the 9 shared sentences."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "ds"
        canonical = make_dataset(root, n_per=4)            # 36 episodes
        r = subprocess.run([sys.executable, str(TOOL), str(root)],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr

        tasks = pq.read_table(root / "meta" / "tasks.parquet").to_pandas()
        assert len(tasks) == 9, f"expected 9 strings, got {len(tasks)}"
        assert list(tasks.task_index) == list(range(9))
        assert set(tasks.task) == set(NATURAL.values())

        data = pq.read_table(root / "data" / "chunk-000" / "file-000.parquet").to_pandas()
        idx_to_task = dict(zip(tasks.task_index, tasks.task))
        for ep, grp in data.groupby("episode_index"):
            assert grp.task_index.nunique() == 1, f"ep {ep} mixed"
            got = idx_to_task[grp.task_index.iloc[0]]
            assert got == NATURAL[canonical[ep]["canonical"]], (ep, got)
        shared = data.groupby("task_index").episode_index.nunique()
        assert shared.min() == 4, dict(shared)
        assert len(data) == 36 * FRAMES, "frames must be untouched"
        assert (root / "meta" / "tasks.parquet.bak").exists()
        print("  36 eps: 36 -> 9 strings, 4 eps each, frames + .bak intact")


def test_refuses_unknown_canonical():
    """An unmapped canonical template must stop, not silently drop episodes."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "ds"
        make_dataset(root, n_per=1)
        c = json.loads((root / "meta" / "canonical_tasks.json").read_text())
        c["99"] = {"canonical": "Do something brand new.", "instruction": "x"}
        (root / "meta" / "canonical_tasks.json").write_text(json.dumps(c))
        r = subprocess.run([sys.executable, str(TOOL), str(root)],
                           capture_output=True, text=True)
        assert r.returncode != 0
        assert "Do something brand new." in r.stdout + r.stderr
        print("  refuses on an unmapped canonical template")


def test_refuses_without_canonical_map():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "ds"
        (root / "meta").mkdir(parents=True)
        r = subprocess.run([sys.executable, str(TOOL), str(root)],
                           capture_output=True, text=True)
        assert r.returncode != 0 and "canonical_tasks.json" in r.stdout + r.stderr
        print("  refuses when canonical_tasks.json is absent")


def test_dry_run_writes_nothing():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "ds"
        make_dataset(root, n_per=2)
        digest = (root / "meta" / "tasks.parquet").read_bytes()
        r = subprocess.run([sys.executable, str(TOOL), str(root), "--dry-run"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert (root / "meta" / "tasks.parquet").read_bytes() == digest
        assert not (root / "meta" / "tasks.parquet.bak").exists()
        print("  --dry-run leaves the dataset byte-identical")


if __name__ == "__main__":
    for fn in (test_natural_map, test_collapse, test_refuses_unknown_canonical,
               test_refuses_without_canonical_map, test_dry_run_writes_nothing):
        print(f"{fn.__name__}:")
        fn()
    print("\nall instruction-collapse checks passed")
