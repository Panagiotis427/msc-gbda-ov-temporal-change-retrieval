"""Deploy the app to its HuggingFace Space (slim, Xet-safe).

Plain ``git push`` to the Space is rejected by HF because the repo history
contains binaries (models/*.pt, demos/*.mp4, figures). This uploads only what the
Space needs, via the ``huggingface_hub`` API, which stores binaries through
Xet/LFS automatically. It ships:

  * ``app.py`` (Space entry point), ``src/`` (the package), ``requirements.txt``
  * ``tests/fixtures/den_tiny`` (the bundled demo corpus the Space runs on)
  * ``scripts/space_readme.md`` uploaded as the Space's ``README.md`` — its YAML
    front-matter is the Space config, kept separate so the GitHub README stays
    front-matter-free

and deliberately EXCLUDES the ~44 MB ``models/`` + ``demos/`` (Space-irrelevant),
the repo ``README.md``, ``report/``, ``data/``, ``local/``, tests, and
``pyproject.toml`` (its cu128 torch pins would break HF's build — the Space
installs plain ``requirements.txt``).

One commit gives the Space a clean slate: files not re-uploaded are removed in the same commit, and
compiled ``.pyc`` files are never uploaded.

Auth: uses your cached ``huggingface-cli login`` token (or the ``HF_TOKEN`` env var).

It refuses to deploy unless the files it ships are committed and the checked-out branch is the tip of its upstream
branch on GitHub (read with ``git ls-remote``; nothing is fetched): the upload replaces the whole Space, so a stale or
dirty tree would silently roll it back. ``--allow-unsynced`` skips the check.

Run from the repo root:
    python scripts/deploy_space.py [--allow-unsynced]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

REPO_ID = "panagiotis427/Open_Vocabulary_Temporal_Change_Retrieval"
# everything _files() and main() read from the working tree
SHIPPED = ("app.py", "requirements.txt", "src", "tests/fixtures", "scripts/space_readme.md")


def _files() -> list:
    """What the Space needs: the entry point, requirements, the package and the bundled fixture -- never a
    compiled file (a .pyc embeds the absolute path it was compiled at)."""
    keep = [Path("app.py"), Path("requirements.txt")]
    for root in (Path("src"), Path("tests/fixtures")):
        keep += [p for p in sorted(root.rglob("*"))
                 if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")]
    return keep


def _git(repo: Path, *args: str) -> tuple[int, str]:
    """``git -C repo *args`` -> (exit code, stdout without trailing whitespace); a missing git counts as a failure."""
    try:
        proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    except OSError:
        return 1, ""
    return proc.returncode, proc.stdout.rstrip()


def _upstream_problem(repo: Path) -> str | None:
    """Why HEAD is not the tip of its upstream branch on the remote, or None when it is. The remote is read with
    ``git ls-remote``, so nothing is fetched and no ref moves."""
    rc, branch = _git(repo, "symbolic-ref", "--quiet", "--short", "HEAD")
    if rc != 0:
        return "HEAD is detached, so there is no upstream branch to compare with"
    remote = _git(repo, "config", "--get", f"branch.{branch}.remote")[1]
    merge = _git(repo, "config", "--get", f"branch.{branch}.merge")[1]
    if not remote or not merge:
        return f"branch {branch} has no upstream to compare with"
    upstream = f"{remote}/{merge.removeprefix('refs/heads/')}"
    rc, listing = _git(repo, "ls-remote", remote, merge)
    tips = [line.split()[0] for line in listing.splitlines() if line.split()[1:] == [merge]]
    if rc != 0 or not tips:
        return f"could not read {upstream} (offline, or the branch is gone), so its tip is unknown"
    tip, head = tips[0], _git(repo, "rev-parse", "HEAD")[1]
    if head == tip:
        return None
    if _git(repo, "merge-base", "--is-ancestor", head, tip)[0] == 0:
        return f"HEAD {head[:12]} is behind {upstream} ({tip[:12]}); pull first"
    if _git(repo, "merge-base", "--is-ancestor", tip, head)[0] == 0:
        return f"HEAD {head[:12]} has commits that {upstream} ({tip[:12]}) lacks; push first"
    return f"HEAD {head[:12]} is not the tip of {upstream} ({tip[:12]}); fetch, then pull or push"


def sync_problems(repo: Path = Path(".")) -> list:
    """Reasons not to deploy ``repo``'s working tree, empty when there are none: uncommitted or untracked changes in
    the shipped files, or HEAD away from the tip of its upstream branch on the remote."""
    if _git(repo, "rev-parse", "--is-inside-work-tree") != (0, "true"):
        return [f"not a git checkout: {repo}"]
    problems = []
    rc, status = _git(repo, "status", "--porcelain", "--untracked-files=normal", "--", *SHIPPED)
    if rc != 0 or status:
        changed = [line[3:] for line in status.splitlines()]
        problems.append("uncommitted or untracked changes in the shipped files: "
                        + ", ".join(changed[:5]) + (" ..." if len(changed) > 5 else ""))
    upstream = _upstream_problem(repo)
    if upstream:
        problems.append(upstream)
    return problems


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(prog="deploy_space.py",
                                     description="Deploy the app to its HuggingFace Space (slim, Xet-safe).")
    parser.add_argument("--allow-unsynced", action="store_true",
                        help="deploy even if the shipped files have uncommitted changes or HEAD is not the tip of "
                             "its upstream branch (checked with git ls-remote; nothing is fetched)")
    args = parser.parse_args(argv)
    if not args.allow_unsynced:
        problems = sync_problems()
        if problems:
            print("refusing to deploy:\n  " + "\n  ".join(problems) + "\n(--allow-unsynced skips this check)",
                  file=sys.stderr)
            return 1
    api = HfApi()
    who = api.whoami().get("name", "?")
    ops = [CommitOperationAdd(path_in_repo=p.as_posix(), path_or_fileobj=str(p)) for p in _files()]
    # the Space's landing page: its own front-matter config + a concise app blurb, as README.md, so the
    # GitHub README stays front-matter-free; uploaded in the same commit, so the config is never missing
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj="scripts/space_readme.md"))
    new = {op.path_in_repo for op in ops}
    stale = [f for f in api.list_repo_files(REPO_ID, repo_type="space")
             if f not in new and f != ".gitattributes"]          # a clean slate, the LFS config kept
    ops += [CommitOperationDelete(path_in_repo=f) for f in stale]
    print(f"Deploying {REPO_ID} as {who}: {len(new)} files, {len(stale)} stale removed")
    api.create_commit(repo_id=REPO_ID, repo_type="space", operations=ops,
                      commit_message="Deploy the app: code, fixture and landing page")
    print(f"Done. Watch the build: https://huggingface.co/spaces/{REPO_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
