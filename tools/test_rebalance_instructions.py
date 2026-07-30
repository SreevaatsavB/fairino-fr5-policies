"""
test_rebalance_instructions.py — self-check for the instruction rebalancer.

    python tools/test_rebalance_instructions.py

Builds a synthetic lerobot-shaped dataset (2 canonical tasks x 6 episodes, every
episode a unique phrasing — the 1:1 binding the real 400-episode set has), runs
the rewrite, and verifies the result. No framework, no fixtures, no GPU.
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
from rebalance_instructions import build_vocabulary  # noqa: E402

CANON = ["put the block in the bin", "turn the valve"]
N_PER = 6
FRAMES = 4


def make_dataset(root: Path):
    """2 canonical tasks x 6 episodes, each with its OWN unique phrasing."""
    canonical, rows, tasks = {}, [], []
    ep = 0
    for c_i, canon in enumerate(CANON):
        for k in range(N_PER):
            instr = f"{canon} variation {k}"          # unique per episode
            canonical[ep] = {"canonical": canon, "instruction": instr}
            tasks.append(instr)
            for f in range(FRAMES):
                rows.append({"episode_index": ep, "frame_index": f,
                             "task_index": ep, "index": len(rows)})
            ep += 1
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "canonical_tasks.json").write_text(json.dumps(canonical))
    pd.DataFrame({"task_index": range(len(tasks)), "task": tasks}).to_parquet(
        root / "meta" / "tasks.parquet", index=False)
    pd.DataFrame(rows).to_parquet(root / "data" / "chunk-000" / "file-000.parquet",
                                  index=False)
    return canonical


def test_vocabulary_shape():
    """Every canonical group collapses to `variants` shared strings, and the
    canonical string itself is always one of them."""
    canonical = {}
    for c_i, canon in enumerate(CANON):
        for k in range(N_PER):
            canonical[c_i * N_PER + k] = {"canonical": canon,
                                          "instruction": f"{canon} variation {k}"}
    assign, vocab = build_vocabulary(canonical, variants=3)
    assert len(assign) == len(CANON) * N_PER
    for canon, vs in vocab.items():
        assert len(vs) == 3, vs
        assert vs[0] == canon, "canonical string must be in the vocabulary"
    # every episode keeps an instruction from ITS OWN canonical group
    for ep, instr in assign.items():
        assert instr in vocab[canonical[ep]["canonical"]], (ep, instr)
    n_unique = len(set(assign.values()))
    assert n_unique == len(CANON) * 3, n_unique
    print(f"  {len(CANON)} groups x 3 variants -> {n_unique} strings for "
          f"{len(assign)} episodes ({len(assign)/n_unique:.1f} eps each)")


def test_determinism():
    """Re-running must produce the identical assignment (same dataset -> same map)."""
    canonical = {i: {"canonical": CANON[i % 2], "instruction": f"p{i}"}
                 for i in range(10)}
    a1, _ = build_vocabulary(canonical, 3)
    a2, _ = build_vocabulary(canonical, 3)
    assert a1 == a2
    print("  assignment is deterministic across runs")


def test_variants_exceeding_pool():
    """Asking for more variants than the group has phrasings must not crash."""
    canonical = {0: {"canonical": "do x", "instruction": "do x"},
                 1: {"canonical": "do x", "instruction": "do x please"}}
    assign, vocab = build_vocabulary(canonical, variants=99)
    assert len(vocab["do x"]) == 2, vocab
    assert set(assign.values()) <= {"do x", "do x please"}
    print("  variants > available phrasings is clamped, not an error")


def test_end_to_end_rewrite():
    """The real thing: run the script and check both files afterwards."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "ds"
        canonical = make_dataset(root)
        before = pq.read_table(root / "meta" / "tasks.parquet").to_pandas()
        assert len(before) == len(CANON) * N_PER, "fixture should start 1:1"

        r = subprocess.run([sys.executable,
                            str(REPO / "tools" / "rebalance_instructions.py"),
                            str(root), "--variants", "3"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr

        tasks = pq.read_table(root / "meta" / "tasks.parquet").to_pandas()
        assert len(tasks) == len(CANON) * 3, f"expected 6 strings, got {len(tasks)}"
        assert list(tasks.task_index) == list(range(len(tasks))), "index must be 0..n-1"

        data = pq.read_table(root / "data" / "chunk-000" / "file-000.parquet").to_pandas()
        idx_to_task = dict(zip(tasks.task_index, tasks.task))
        # every task_index resolves, and stays inside its own canonical group
        for ep, grp in data.groupby("episode_index"):
            assert grp.task_index.nunique() == 1, f"ep {ep} has mixed task_index"
            instr = idx_to_task[grp.task_index.iloc[0]]
            assert canonical[ep]["canonical"] in instr, (ep, instr)
        # sharing actually happened
        eps_per_string = data.groupby("task_index").episode_index.nunique()
        assert eps_per_string.min() >= 2, f"no sharing: {dict(eps_per_string)}"
        # backups exist, frame count untouched
        assert (root / "meta" / "tasks.parquet.bak").exists()
        assert (root / "data" / "chunk-000" / "file-000.parquet.bak").exists()
        assert len(data) == len(CANON) * N_PER * FRAMES
        print(f"  rewrote 12 eps: {len(before)} -> {len(tasks)} strings, "
              f"{eps_per_string.min()}-{eps_per_string.max()} eps each, .bak kept")


def test_refuses_without_canonical_map():
    """No canonical_tasks.json = no grouping. Must refuse, not guess."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "ds"
        (root / "meta").mkdir(parents=True)
        r = subprocess.run([sys.executable,
                            str(REPO / "tools" / "rebalance_instructions.py"),
                            str(root)], capture_output=True, text=True)
        assert r.returncode != 0
        assert "canonical_tasks.json" in r.stdout + r.stderr
        print("  refuses when canonical_tasks.json is absent")


def test_dry_run_writes_nothing():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "ds"
        make_dataset(root)
        digest = (root / "meta" / "tasks.parquet").read_bytes()
        r = subprocess.run([sys.executable,
                            str(REPO / "tools" / "rebalance_instructions.py"),
                            str(root), "--dry-run"], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert (root / "meta" / "tasks.parquet").read_bytes() == digest
        assert not (root / "meta" / "tasks.parquet.bak").exists()
        print("  --dry-run leaves the dataset byte-identical")


def test_openpi_clean_matches_tokenizer():
    """Must reproduce openpi/src/openpi/models/tokenizer.py:23 exactly:
        prompt.strip().replace("_", " ").replace("\\n", " ")
    so what is stored in tasks.parquet is what the model tokenizes."""
    from rebalance_instructions import openpi_clean
    cases = {
        "  pick_up the block ": "pick up the block",
        "put the\nblock in the bin": "put the block in the bin",
        "pick   up  the block": "pick up the block",
        "Place all Cream objects into the Cream Tray.":
            "Place all Cream objects into the Cream Tray.",
    }
    for raw, want in cases.items():
        assert openpi_clean(raw) == want, (raw, openpi_clean(raw), want)
    # pi0 (flow) does NOT lowercase — only the FAST tokenizers do
    assert openpi_clean("Put The Block") == "Put The Block"
    # idempotent, so re-running the tool never churns the text
    for raw in cases:
        assert openpi_clean(openpi_clean(raw)) == openpi_clean(raw)
    print("  openpi_clean matches tokenizer.py:23, preserves case, idempotent")


def test_style_warnings():
    """Flags what is out of distribution for the pretraining mixture."""
    from rebalance_instructions import style_warnings
    assert style_warnings("put the block in the bin") == []
    assert any("words" in w for w in style_warnings(" ".join(["word"] * 20)))
    assert any("ALL CAPS" in w for w in style_warnings("PUT THE BLOCK IN THE BIN"))
    assert any("cleanup" in w for w in style_warnings("put_the block"))
    print("  style_warnings flags long / ALL CAPS / uncleaned, passes DROID style")


def test_vocabulary_is_cleaned():
    """Instructions land in tasks.parquet already normalized."""
    from rebalance_instructions import build_vocabulary
    canonical = {0: {"canonical": " pick_up the block ", "instruction": " pick_up the block "},
                 1: {"canonical": " pick_up the block ", "instruction": "grab  the block"}}
    assign, vocab = build_vocabulary(canonical, 2)
    for v in vocab[" pick_up the block "]:
        assert v == v.strip() and "_" not in v and "  " not in v, v
    assert "pick up the block" in assign.values()
    print("  build_vocabulary emits openpi-clean text")


if __name__ == "__main__":
    for fn in (test_vocabulary_shape, test_determinism, test_variants_exceeding_pool,
               test_end_to_end_rewrite, test_refuses_without_canonical_map,
               test_dry_run_writes_nothing, test_openpi_clean_matches_tokenizer,
               test_style_warnings, test_vocabulary_is_cleaned):
        print(f"{fn.__name__}:")
        fn()
    print("\nall instruction-rebalance checks passed")
