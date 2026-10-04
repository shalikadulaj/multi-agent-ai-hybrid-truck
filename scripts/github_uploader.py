#!/usr/bin/env python3
"""Helper script to prepare and push this repository to GitHub.

Usage examples:
    python scripts/github_uploader.py --repo-url https://github.com/<user>/<repo>.git --branch main
    python scripts/github_uploader.py --repo-url https://github.com/<user>/<repo>.git --branch main --message "Hybrid truck multi-agent EMS"
    python scripts/github_uploader.py --dry-run

Notes:
- This script does not create a GitHub account or authenticate for you.
- You still need to have GitHub credentials configured on your machine.
- If the repo is not yet initialized, git will initialize it automatically.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_command(command: list[str], cwd: Path = ROOT) -> str:
    print(f"$ {' '.join(command)}")
    result = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}")
    return result.stdout.strip()


def git_repo_exists() -> bool:
    try:
        run_command(["git", "rev-parse", "--show-toplevel"], cwd=ROOT)
        return True
    except RuntimeError:
        return False


def ensure_repo() -> None:
    if git_repo_exists():
        return

    run_command(["git", "init"], cwd=ROOT)
    print("Initialized a new git repository.")


def add_remote(repo_url: str) -> None:
    try:
        run_command(["git", "remote", "get-url", "origin"], cwd=ROOT)
        current = run_command(["git", "remote", "get-url", "origin"], cwd=ROOT)
        if current.strip() != repo_url:
            print(f"Origin already exists as: {current}")
            print("Use a different remote or update it manually before pushing.")
    except RuntimeError:
        run_command(["git", "remote", "add", "origin", repo_url], cwd=ROOT)
        print(f"Added GitHub remote: {repo_url}")


def stage_and_commit(branch: str, message: str) -> None:
    run_command(["git", "checkout", "-B", branch], cwd=ROOT)
    run_command(["git", "add", "."], cwd=ROOT)
    try:
        run_command(["git", "commit", "-m", message], cwd=ROOT)
    except RuntimeError:
        print("No changes to commit or git user not configured yet.")


def push_to_remote(branch: str) -> None:
    run_command(["git", "push", "-u", "origin", branch], cwd=ROOT)
    print("Repository pushed successfully.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and push this project to GitHub.")
    parser.add_argument("--repo-url", help="GitHub remote URL, for example: https://github.com/<user>/<repo>.git")
    parser.add_argument("--branch", default="main", help="Branch name to push (default: main)")
    parser.add_argument("--message", default="Add multi-agent hybrid truck EMS project", help="Git commit message")
    parser.add_argument("--dry-run", action="store_true", help="Show the planned actions without executing them")
    args = parser.parse_args()

    if args.dry_run:
        print("Dry run enabled. No git commands will be executed.")
        print(f"Repository root: {ROOT}")
        print(f"Target branch: {args.branch}")
        print(f"Remote URL: {args.repo_url}")
        return 0

    ensure_repo()

    if args.repo_url:
        add_remote(args.repo_url)

    stage_and_commit(args.branch, args.message)

    if args.repo_url:
        push_to_remote(args.branch)
    else:
        print("No remote URL was provided. Repository is ready for manual push.")
        print("Example: git remote add origin https://github.com/<user>/<repo>.git")
        print("Then: git push -u origin main")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
