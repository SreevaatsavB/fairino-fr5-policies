#!/usr/bin/env python3
"""
rebalance_instructions.py — shrink a dataset's instruction vocabulary so language
carries task meaning instead of episode identity.

The problem
───────────
convert_episodes.py passes each raw episode's `language_instruction` through
unchanged, so the 2026-07 set ends up with 400 unique strings over 9 canonical
tasks — a 1:1 episode↔phrasing binding. The instruction then works as an
episode-ID lookup key: a model can use it to recall "which trajectory was this"
rather than "what am I being asked to do". The v3 training patch papered over
this by swapping in the canonical string 50% of the time; this fixes the data.

What it does
────────────
For each canonical task, keeps a SMALL SHARED vocabulary — the canonical string
plus (variants-1) of that group's existing phrasings — and assigns it round-robin
across the group's episodes. Nothing is invented: every surviving instruction is
one the operator actually wrote.

    before   400 strings / 400 episodes   = 1.0 episodes per string
    after      9 x 5 = 45 strings         ~ 8.9 episodes per string

Language still separates the 9 canonical tasks (real signal) and still varies in
phrasing (paraphrase robustness), but can no longer identify an episode.

Usage
─────
    python tools/rebalance_instructions.py <dataset_root> --variants 5 --dry-run
    python tools/rebalance_instructions.py <dataset_root> --variants 5

Rewrites meta/tasks.parquet and the data parquet's task_index column in place
(after a .bak copy). Requires meta/canonical_tasks.json (written by
convert_and_push_dataset_v2.ipynb §8); without it there is no grouping to
rebalance and the script refuses rather than guessing.
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def build_vocabulary(canonical: dict, variants: int):
    """episode -> new instruction, using a small shared vocabulary per canonical task.

    canonical: {ep_index: {"canonical": str, "instruction": str}}
    Deterministic: groups and phrasings are sorted before selection, so re-running
    on the same dataset produces the same assignment.
    """
    groups = defaultdict(list)
    for ep, rec in sorted(canonical.items(), key=lambda kv: int(kv[0])):
        groups[rec["canonical"]].append((int(ep), rec["instruction"]))

    assignment, vocab_per_group = {}, {}
    for canon, members in sorted(groups.items()):
        # the canonical string first, then distinct existing phrasings, sorted for
        # determinism. Never invents text.
        pool = [canon] + [p for p in sorted({m[1] for m in members}) if p != canon]
        vocab = pool[:max(1, variants)]
        vocab_per_group[canon] = vocab
        for i, (ep, _) in enumerate(members):
            assignment[ep] = vocab[i % len(vocab)]
    return assignment, vocab_per_group


def _stats(per_string_counts):
    n_str = len(per_string_counts)
    n_eps = sum(per_string_counts.values())
    return n_str, n_eps, (n_eps / n_str if n_str else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="lerobot dataset root (contains meta/ and data/)")
    ap.add_argument("--variants", type=int, default=5,
                    help="phrasings kept per canonical task (default 5)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the change without writing anything")
    args = ap.parse_args()

    root = Path(args.root)
    canon_path = root / "meta" / "canonical_tasks.json"
    if not canon_path.exists():
        raise SystemExit(
            f"{canon_path} not found. It maps episode -> canonical task and is what "
            f"defines the grouping; without it there is nothing to rebalance. It is "
            f"written by convert_and_push_dataset_v2.ipynb section 8.")

    canonical = json.loads(canon_path.read_text())
    assignment, vocab = build_vocabulary(canonical, args.variants)

    before = defaultdict(int)
    for rec in canonical.values():
        before[rec["instruction"]] += 1
    after = defaultdict(int)
    for instr in assignment.values():
        after[instr] += 1

    b, n_eps, b_ratio = _stats(before)
    a, _, a_ratio = _stats(after)
    print(f"episodes                : {n_eps}")
    print(f"canonical tasks         : {len(vocab)}")
    print(f"unique instructions     : {b} -> {a}")
    print(f"episodes per instruction: {b_ratio:.1f} -> {a_ratio:.1f}")
    print()
    for canon, vs in sorted(vocab.items()):
        print(f"  [{canon}]")
        for v in vs:
            print(f"      {after[v]:3d} eps  {v}")

    if args.dry_run:
        print("\ndry run — nothing written")
        return

    # ── rewrite meta/tasks.parquet ────────────────────────────────────────────
    new_tasks = sorted(after)                       # deterministic index order
    task_to_idx = {t: i for i, t in enumerate(new_tasks)}
    tasks_path = root / "meta" / "tasks.parquet"
    shutil.copy2(tasks_path, tasks_path.with_suffix(".parquet.bak"))
    pd.DataFrame({"task_index": range(len(new_tasks)),
                  "task": new_tasks}).to_parquet(tasks_path, index=False)

    # ── rewrite task_index in every data shard ────────────────────────────────
    shards = sorted((root / "data").rglob("*.parquet"))
    if not shards:
        raise SystemExit(f"no data parquet under {root/'data'}")
    for shard in shards:
        df = pq.read_table(shard).to_pandas()
        missing = set(df["episode_index"].unique()) - set(assignment)
        if missing:
            raise SystemExit(f"{shard}: episodes {sorted(missing)[:5]} absent from "
                             f"canonical_tasks.json — refusing a partial rewrite")
        shutil.copy2(shard, shard.with_suffix(".parquet.bak"))
        df["task_index"] = df["episode_index"].map(
            lambda e: task_to_idx[assignment[int(e)]]).astype("int64")
        df.to_parquet(shard, index=False)
        print(f"rewrote {shard.relative_to(root)} ({len(df)} rows)")

    print(f"\nwrote {tasks_path.relative_to(root)} ({len(new_tasks)} instructions)")
    print("originals kept as *.bak next to each rewritten file")


if __name__ == "__main__":
    main()
