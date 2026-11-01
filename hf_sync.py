#!/usr/bin/env python3
"""Sync RASM *large assets* (datasets, model weights, result images) with
HuggingFace Datasets.

The GitHub repo only stores code/docs (see github_sync.py). Everything big
is kept on HuggingFace to keep the git repo small and cloneable.

Usage
-----
  # Upload local big assets to HuggingFace
  python hf_sync.py push --token <HF_TOKEN>

  # Download big assets back into this repo
  python hf_sync.py pull --token <HF_TOKEN>

You can also provide the token via the environment variable HF_TOKEN.

Requires:  pip install huggingface_hub

The HuggingFace dataset repo used is `oiu996/rasm-data` (created automatically
on first push if it does not exist yet).
"""
import argparse
import os
import sys

try:
    from huggingface_hub import HfApi, upload_folder, snapshot_download
except ImportError:
    print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

REPO_ID = "oiuuu/rasm-data"
REPO_TYPE = "dataset"

# Folders that are externalized (kept out of git, stored on HuggingFace).
LOCAL_DIRS = [
    "datasets",
    "log",
    "runs",
    "results_istd",
    "results_srd",
    "results_istd_on_srd",
    "results_srd_on_istd",
    "showcase",
]


def push(token):
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: no HF token. Use --token or set HF_TOKEN.")
        sys.exit(1)
    api = HfApi(token=token)
    api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, exist_ok=True)
    for d in LOCAL_DIRS:
        if os.path.isdir(d):
            print(f"[upload] {d} -> {REPO_ID}")
            upload_folder(
                folder_path=d,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                token=token,
            )
        else:
            print(f"[skip]   not found: {d}")


def pull(token):
    token = token or os.environ.get("HF_TOKEN")
    print(f"[download] {REPO_ID} -> ./")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=".",
        token=token,
        allow_patterns=[
            "datasets/**",
            "log/**",
            "runs/**",
            "results_istd/**",
            "results_srd/**",
            "results_istd_on_srd/**",
            "results_srd_on_istd/**",
            "showcase/**",
        ],
    )


def main():
    p = argparse.ArgumentParser(description="Sync RASM assets with HuggingFace")
    p.add_argument("action", choices=["push", "pull"])
    p.add_argument("--token", default=None, help="HuggingFace token (or set HF_TOKEN)")
    args = p.parse_args()

    if args.action == "push":
        push(args.token)
    else:
        pull(args.token)


if __name__ == "__main__":
    main()
