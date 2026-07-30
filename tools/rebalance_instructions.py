#!/usr/bin/env python3
"""
rebalance_instructions.py — collapse the instruction vocabulary to 9 shared,
natural sentences so language means "which colour goes in which tray", nothing else.

Why: the 2026-07 set carries 400 unique per-episode instructions over 9 canonical
tasks, so the instruction identifies the episode. Counts and shapes ("the three
brown cubes", "1 cube + 1 cuboid") belong to the camera, not the prompt — leaving
them in the text lets the model read them instead of looking.

    before  400 strings / 400 episodes  = 1.0 episodes per string
    after     9 strings / 400 episodes  ~  44 episodes per string

Usage
─────
    python tools/rebalance_instructions.py <dataset_root> --dry-run
    python tools/rebalance_instructions.py <dataset_root>

Rewrites meta/tasks.parquet and every data shard's task_index (.bak alongside).
Needs meta/canonical_tasks.json for the episode -> canonical mapping.
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# The 9 canonical templates, rewritten as plain spoken English — colour + tray only.
#
# Form mirrors a real DROID-Kitchen label ("pick up the sponge and put it in the
# sink"), the closest match in pi0's pretraining mixture. Four grammar decisions,
# none of them cosmetic:
#   - "each blue block ... put it": distributive singular, so the sentence is
#     grammatical whether the scene holds one object or four. 146 of 400 episodes
#     have exactly one, so a bare plural ("the blue blocks ... put them") would be
#     ungrammatical for over a third of the dataset.
#   - sentence case + full stop, matching the source canonical templates and DROID's
#     written labels. openpi's tokenizer does NOT lowercase (only the FAST ones do),
#     so what is written here is what the model sees.
#   - "wooden tray": attributive adjective, not the noun-adjunct "wood tray".
#   - no counts, no shapes (cube vs cuboid). Those are in the camera; naming them in
#     the prompt lets the model read instead of look.
NATURAL = {
    "Place all Blue objects into the Brown Tray.":
        "Pick up each blue block and put it in the brown tray.",
    "Place all Blue objects into the Cream Tray.":
        "Pick up each blue block and put it in the cream tray.",
    "Place all Blue objects into the Wood Tray.":
        "Pick up each blue block and put it in the wooden tray.",
    "Place all Brown objects into the Brown Tray.":
        "Pick up each brown block and put it in the brown tray.",
    "Place all Brown objects into the Cream Tray.":
        "Pick up each brown block and put it in the cream tray.",
    "Place all Brown objects into the Wood Tray.":
        "Pick up each brown block and put it in the wooden tray.",
    "Place all Cream objects into the Brown Tray.":
        "Pick up each cream block and put it in the brown tray.",
    "Place all Cream objects into the Cream Tray.":
        "Pick up each cream block and put it in the cream tray.",
    "Place all Cream objects into the Wood Tray.":
        "Pick up each cream block and put it in the wooden tray.",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="lerobot dataset root (contains meta/ and data/)")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = ap.parse_args()

    root = Path(args.root)
    canon_path = root / "meta" / "canonical_tasks.json"
    if not canon_path.exists():
        raise SystemExit(f"{canon_path} not found — it maps episode -> canonical task "
                         f"(written by convert_and_push_dataset_v2.ipynb section 8)")
    canonical = json.loads(canon_path.read_text())

    unknown = {v["canonical"] for v in canonical.values()} - set(NATURAL)
    if unknown:
        raise SystemExit("canonical templates with no natural rewrite — add them to "
                         "NATURAL:\n  " + "\n  ".join(sorted(unknown)))

    assignment = {int(ep): NATURAL[v["canonical"]] for ep, v in canonical.items()}
    counts = Counter(assignment.values())
    before = len({v["instruction"] for v in canonical.values()})

    print(f"episodes            : {len(assignment)}")
    print(f"unique instructions : {before} -> {len(counts)}")
    print(f"episodes per string : {len(assignment)/max(before,1):.2f} -> "
          f"{len(assignment)/len(counts):.1f}\n")
    for text, n in sorted(counts.items()):
        print(f"  {n:4d} eps   {text}")

    if args.dry_run:
        print("\ndry run — nothing written")
        return

    tasks = sorted(counts)
    idx = {t: i for i, t in enumerate(tasks)}
    tp = root / "meta" / "tasks.parquet"
    shutil.copy2(tp, tp.with_suffix(".parquet.bak"))
    pd.DataFrame({"task_index": range(len(tasks)), "task": tasks}).to_parquet(tp, index=False)

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
            lambda e: idx[assignment[int(e)]]).astype("int64")
        df.to_parquet(shard, index=False)
        print(f"rewrote {shard.relative_to(root)} ({len(df)} rows)")
    print(f"\nwrote {tp.relative_to(root)} ({len(tasks)} instructions), .bak kept")


if __name__ == "__main__":
    main()
