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

Run from the repo root:
    python scripts/deploy_space.py
"""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

REPO_ID = "panagiotis427/Open_Vocabulary_Temporal_Change_Retrieval"


def _files() -> list:
    """What the Space needs: the entry point, requirements, the package and the bundled fixture -- never a
    compiled file (a .pyc embeds the absolute path it was compiled at)."""
    keep = [Path("app.py"), Path("requirements.txt")]
    for root in (Path("src"), Path("tests/fixtures")):
        keep += [p for p in sorted(root.rglob("*"))
                 if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")]
    return keep


def main() -> None:
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


if __name__ == "__main__":
    main()
