"""
tools/push_dataset_hf.py — one-time upload of the FR5 LeRobot dataset to the
HuggingFace Hub, so cloud training pods (RunPod, see notebooks/) can pull it with
snapshot_download instead of manual scp/zip juggling.

Run wherever the full dataset lives (GPU box or laptop):

    huggingface-cli login          # token needs WRITE access
    python tools/push_dataset_hf.py \
        --root lerobot_dataset \
        --repo <hf-username>/fr5-pick-place-lerobot

The repo is created PRIVATE by default. Re-running uploads only changed files
(hub-side deduplication), so it doubles as a sync command after re-recording.
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="lerobot_dataset",
                    help="path to the LeRobot dataset directory (contains meta/, data/, ...)")
    ap.add_argument("--repo", required=True,
                    help="HF dataset repo id, e.g. myuser/fr5-pick-place-lerobot")
    ap.add_argument("--public", action="store_true",
                    help="create the repo public (default: private)")
    args = ap.parse_args()

    root = Path(args.root)
    if not (root / "meta" / "info.json").exists():
        raise SystemExit(f"{root} does not look like a LeRobot dataset "
                         f"(missing meta/info.json)")

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=not args.public,
                    exist_ok=True)
    print(f"uploading {root} -> hf://datasets/{args.repo} "
          f"({'public' if args.public else 'private'}) ...")
    api.upload_folder(folder_path=str(root), repo_id=args.repo,
                      repo_type="dataset",
                      commit_message=f"sync from {root.resolve().name}")
    print("done — pull it on a pod with:")
    print(f'  huggingface_hub.snapshot_download("{args.repo}", repo_type="dataset", '
          f'local_dir="lerobot_dataset")')


if __name__ == "__main__":
    main()
