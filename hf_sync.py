# #!/usr/bin/env python3
# """Sync RASM *large assets* (datasets, model weights, result images) with
# HuggingFace Datasets.

# The GitHub repo only stores code/docs (see github_sync.py). Everything big
# is kept on HuggingFace to keep the git repo small and cloneable.

# Usage
# -----
#   # Upload local big assets to HuggingFace
#   python hf_sync.py push --token <HF_TOKEN>

#   # Download big assets back into this repo
#   python hf_sync.py pull --token <HF_TOKEN>

# You can also provide the token via the environment variable HF_TOKEN.

# Requires:  pip install huggingface_hub

# The HuggingFace dataset repo used is `oiu996/rasm-data` (created automatically
# on first push if it does not exist yet).
# """
# import argparse
# import os
# import sys

# try:
#     from huggingface_hub import HfApi, upload_folder, snapshot_download
# except ImportError:
#     print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
#     sys.exit(1)

# REPO_ID = "oiuuu/rasm-data"
# REPO_TYPE = "dataset"

# # Folders that are externalized (kept out of git, stored on HuggingFace).
# # Note: result images (results_*) and showcase/ are now tracked in git,
# # so only raw datasets / weights / run logs remain here.
# LOCAL_DIRS = [
#     "datasets",
#     "log",
#     "runs",
# ]


# def push(token):
#     token = token or os.environ.get("HF_TOKEN")
#     if not token:
#         print("ERROR: no HF token. Use --token or set HF_TOKEN.")
#         sys.exit(1)
#     api = HfApi(token=token)
#     api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, exist_ok=True)
#     for d in LOCAL_DIRS:
#         if os.path.isdir(d):
#             print(f"[upload] {d} -> {REPO_ID}")
#             upload_folder(
#                 folder_path=d,
#                 repo_id=REPO_ID,
#                 repo_type=REPO_TYPE,
#                 token=token,
#             )
#         else:
#             print(f"[skip]   not found: {d}")


# def pull(token):
#     token = token or os.environ.get("HF_TOKEN")
#     print(f"[download] {REPO_ID} -> ./")
#     snapshot_download(
#         repo_id=REPO_ID,
#         repo_type=REPO_TYPE,
#         local_dir=".",
#         token=token,
#         allow_patterns=[
#             "datasets/**",
#             "log/**",
#             "runs/**",
#         ],
#     )


# def main():
#     p = argparse.ArgumentParser(description="Sync RASM assets with HuggingFace")
#     p.add_argument("action", choices=["push", "pull"])
#     p.add_argument("--token", default=None, help="HuggingFace token (or set HF_TOKEN)")
#     args = p.parse_args()

#     if args.action == "push":
#         push(args.token)
#     else:
#         pull(args.token)


# if __name__ == "__main__":
#     main()
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
Optional:  pip install hf_transfer       # faster, lower-memory uploads

The HuggingFace dataset repo used is `oiuuu/rasm-data` (created automatically
on first push if it does not exist yet).
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from huggingface_hub import HfApi, snapshot_download
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
]

# Directories / files that should never be uploaded.
_IGNORED_DIR_NAMES = {
    ".cache", "_tmp_episodes", "__pycache__", ".ipynb_checkpoints",
}
_IGNORED_FILE_SUFFIXES = (".tmp", ".partial", ".lock", ".swp")
_IGNORED_FILE_NAMES = {"launch_nohup.log", ".DS_Store", "Thumbs.db"}

# Enable hf_transfer (Rust backend) if available: lower memory + faster.
try:
    import hf_transfer  # noqa: F401
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
except ImportError:
    pass


def _should_skip_file(name):
    if name in _IGNORED_FILE_NAMES:
        return True
    return name.endswith(_IGNORED_FILE_SUFFIXES)


def _iter_files(root_dir, suffix_filter=None):
    """Yield every file under root_dir, skipping ignored dirs/files.

    If `suffix_filter` is given (e.g. (".zip",)), only files whose name ends
    with one of those suffixes are yielded.
    """
    if suffix_filter is not None:
        suffix_filter = tuple(s.lower() for s in suffix_filter)
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in _IGNORED_DIR_NAMES]
        for fname in files:
            if _should_skip_file(fname):
                continue
            if suffix_filter is not None and not fname.lower().endswith(suffix_filter):
                continue
            yield os.path.join(root, fname)


def _repo_rel_path(fpath, top_level_dir):
    """`datasets/a/b.txt` (relative to repo root) for `top_level_dir='datasets'`."""
    parent = os.path.dirname(os.path.abspath(top_level_dir))
    return os.path.relpath(os.path.abspath(fpath), parent).replace(os.sep, "/")


def _upload_one(api, fpath, repo_rel, token):
    api.upload_file(
        path_or_fileobj=fpath,
        path_in_repo=repo_rel,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        token=token,
    )


def _push_dir(api, top_level_dir, token, max_workers=4, retries=2, suffix_filter=None):
    """Upload one top-level directory, keeping the `<dir>/...` prefix in the repo.

    If `suffix_filter` is set, only matching files are uploaded (e.g. datasets
    pushes only the .zip archives, not the extracted image folders).
    """
    files = list(_iter_files(top_level_dir, suffix_filter=suffix_filter))
    total = len(files)
    if total == 0:
        print(f"  [empty] {top_level_dir}")
        return []

    print(f"  {total} files to upload under {top_level_dir}/  "
          f"(workers={max_workers})")

    failed = []
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _upload_one, api, f, _repo_rel_path(f, top_level_dir), token
            ): f
            for f in files
        }
        for fut in as_completed(futures):
            fpath = futures[fut]
            done += 1
            try:
                fut.result()
            except Exception as e:
                # One retry per failed file before giving up on it.
                err = str(e)
                for _ in range(retries):
                    try:
                        _upload_one(api, fpath,
                                    _repo_rel_path(fpath, top_level_dir), token)
                        err = None
                        break
                    except Exception as e2:
                        err = str(e2)
                if err is not None:
                    failed.append((fpath, err))
                    print(f"  [FAIL] {fpath}: {err}")
            if done % 100 == 0 or done == total:
                print(f"  [progress] {done}/{total}  (failed: {len(failed)})")
    return failed


def push(token, workers=4):
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: no HF token. Use --token or set HF_TOKEN.")
        sys.exit(1)

    api = HfApi(token=token)
    api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, exist_ok=True)

    all_failed = []
    for d in LOCAL_DIRS:
        if not os.path.isdir(d):
            print(f"[skip]   not found: {d}")
            continue
        if d == "datasets":
            # Only upload the raw .zip archives (ISTD_ABC.zip, SRD_ABC.zip);
            # the extracted ISTD_Dataset/ and SRD_ABC/ folders are skipped.
            print(f"[upload] {d} (zip archives only) -> {REPO_ID}/{d}")
            failed = _push_dir(
                api, d, token, max_workers=workers, suffix_filter=(".zip",)
            )
        else:
            print(f"[upload] {d} -> {REPO_ID}/{d}")
            failed = _push_dir(api, d, token, max_workers=workers)
        all_failed.extend((d, f, err) for f, err in failed)

    if all_failed:
        print(f"\n{len(all_failed)} file(s) failed to upload:")
        for _d, f, err in all_failed[:20]:
            print(f"  - {f}: {err}")
        if len(all_failed) > 20:
            print(f"  ... and {len(all_failed) - 20} more")
        sys.exit(2)


def pull(token):
    token = token or os.environ.get("HF_TOKEN")
    print(f"[download] {REPO_ID} -> ./")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=".",
        token=token,
        allow_patterns=[
            "datasets/**/*.zip",
            "log/**",
            "runs/**",
        ],
    )


def main():
    p = argparse.ArgumentParser(description="Sync RASM assets with HuggingFace")
    p.add_argument("action", choices=["push", "pull"])
    p.add_argument("--token", default=None,
                   help="HuggingFace token (or set HF_TOKEN)")
    p.add_argument("--workers", type=int, default=4,
                   help="Concurrent upload workers (default: 4)")
    args = p.parse_args()

    if args.action == "push":
        push(args.token, workers=args.workers)
    else:
        pull(args.token)


if __name__ == "__main__":
    main()