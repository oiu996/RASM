#!/usr/bin/env python3
"""Sync RASM *source code* with the GitHub repository.

This script only handles the lightweight code/docs (the parts that live in
git). Large assets (datasets, model weights, result images) are synced
separately with `hf_sync.py` to HuggingFace.

Usage
-----
  # Push local changes to GitHub
  python github_sync.py push --token <GITHUB_PAT> --message "update" --branch master

  # Pull latest changes from GitHub
  python github_sync.py pull --token <GITHUB_PAT> --branch master

You can also provide the token via the environment variable GITHUB_TOKEN
instead of the --token argument.

The token must be a GitHub Personal Access Token with the `repo` scope.
The remote is https://github.com/oiu996/RASM.git ; the token is injected
into the URL only for the duration of the push/pull and then removed.
"""
import argparse
import os
import subprocess
import sys

REPO = "oiu996/RASM"
REMOTE_URL = f"https://github.com/{REPO}.git"


def run(cmd, check=True):
    print("+ " + " ".join(cmd))
    r = subprocess.run(cmd, check=check)
    return r


def auth_url(token):
    return f"https://{token}@github.com/{REPO}.git"


def ensure_remote():
    out = subprocess.run(["git", "remote", "get-url", "origin"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        run(["git", "remote", "add", "origin", REMOTE_URL])


def push(token, message, branch):
    token = token or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: no GitHub token. Use --token or set GITHUB_TOKEN.")
        sys.exit(1)
    ensure_remote()
    run(["git", "remote", "set-url", "origin", auth_url(token)])
    try:
        run(["git", "add", "-A"])
        status = subprocess.run(["git", "status", "--porcelain"],
                                capture_output=True, text=True)
        if status.stdout.strip():
            msg = message or "Update RASM code"
            run(["git", "commit", "-m", msg])
        else:
            print("No changes to commit.")
        run(["git", "push", "-u", "origin", branch])
    finally:
        run(["git", "remote", "set-url", "origin", REMOTE_URL])


def pull(token, branch):
    token = token or os.environ.get("GITHUB_TOKEN")
    ensure_remote()
    if token:
        run(["git", "remote", "set-url", "origin", auth_url(token)])
    try:
        run(["git", "pull", "origin", branch])
    finally:
        run(["git", "remote", "set-url", "origin", REMOTE_URL])


def main():
    p = argparse.ArgumentParser(description="Sync RASM code with GitHub")
    p.add_argument("action", choices=["push", "pull"])
    p.add_argument("--token", default=None, help="GitHub PAT (or set GITHUB_TOKEN)")
    p.add_argument("--message", default=None, help="commit message for push")
    p.add_argument("--branch", default="master")
    args = p.parse_args()

    if args.action == "push":
        push(args.token, args.message, args.branch)
    else:
        pull(args.token, args.branch)


if __name__ == "__main__":
    main()
