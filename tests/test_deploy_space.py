"""Offline tests for the guard in scripts/deploy_space.py: throwaway git repositories with local bare remotes, and a
stub in place of the Hugging Face API, so nothing reaches the network."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import deploy_space as ds


@pytest.fixture
def hermetic_git(tmp_path, monkeypatch):
    """Git for these tests reads only this config: no system or user hooks, signing or line-ending rewrites."""
    cfg = tmp_path / "gitconfig"
    cfg.write_text("[init]\n\tdefaultBranch = main\n[user]\n\tname = t\n\temail = t@example.invalid\n"
                   "[commit]\n\tgpgsign = false\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(cfg))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    return tmp_path


def _g(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _commit(work: Path, files: dict) -> None:
    for name, content in files.items():
        (work / name).parent.mkdir(parents=True, exist_ok=True)
        (work / name).write_text(content, encoding="utf-8")
    _g(work, "add", "-A")
    _g(work, "commit", "-q", "-m", "update")


def _published(root: Path) -> tuple[Path, Path]:
    """A bare remote and a clone whose main tracks it, holding every shipped path plus a README the Space never
    receives; HEAD is the remote's tip."""
    remote, work = root / "repo.git", root / "repo"
    _g(root, "init", "-q", "--bare", str(remote))
    _g(root, "clone", "-q", str(remote), str(work))
    _commit(work, {"app.py": "app", "requirements.txt": "gradio", "src/m.py": "x = 1",
                   "tests/fixtures/d.txt": "fixture", "scripts/space_readme.md": "card", "README.md": "docs"})
    _g(work, "push", "-q", "-u", "origin", "main")
    return remote, work


def test_sync_check_passes_for_a_clean_tree_at_its_remote_tip(hermetic_git):
    _, work = _published(hermetic_git)
    assert ds.sync_problems(work) == []


def test_sync_check_flags_changes_in_shipped_files_only(hermetic_git):
    _, work = _published(hermetic_git)
    (work / "README.md").write_text("edited docs", encoding="utf-8")       # never uploaded: no reason to refuse
    assert ds.sync_problems(work) == []
    (work / "src" / "m.py").write_text("x = 2", encoding="utf-8")
    (work / "src" / "new.py").write_text("y = 1", encoding="utf-8")        # untracked, but the upload would ship it
    [problem] = ds.sync_problems(work)
    assert problem.startswith("uncommitted or untracked changes in the shipped files: ")
    assert "src/m.py" in problem and "src/new.py" in problem


def test_sync_check_flags_a_tree_behind_or_ahead_of_its_remote(hermetic_git):
    remote, work = _published(hermetic_git)
    other = hermetic_git / "other"
    _g(hermetic_git, "clone", "-q", str(remote), str(other))
    _commit(other, {"src/m.py": "x = 3"})
    _g(other, "push", "-q")
    [problem] = ds.sync_problems(work)                     # the new tip is not fetched, so its relation is unknown
    assert "is not the tip of origin/main" in problem
    _g(work, "fetch", "-q")
    [problem] = ds.sync_problems(work)
    assert "is behind origin/main" in problem and problem.endswith("pull first")
    _g(work, "merge", "-q", "--ff-only", "origin/main")
    _commit(work, {"src/m.py": "x = 4"})
    [problem] = ds.sync_problems(work)
    assert "has commits that origin/main" in problem and problem.endswith("push first")


def test_sync_check_needs_a_checkout_and_an_upstream(hermetic_git):
    assert ds.sync_problems(hermetic_git / "nowhere") == [f"not a git checkout: {hermetic_git / 'nowhere'}"]
    _, work = _published(hermetic_git)
    _g(work, "checkout", "-q", "-b", "local-only")
    assert ds.sync_problems(work) == ["branch local-only has no upstream to compare with"]


def test_main_refuses_before_touching_the_hub_unless_allowed(monkeypatch, capsys):
    calls = []

    class FakeApi:
        def whoami(self):
            return {"name": "tester"}

        def list_repo_files(self, repo_id, repo_type):
            return ["old.py", ".gitattributes"]

        def create_commit(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(ds, "HfApi", FakeApi)
    monkeypatch.setattr(ds, "sync_problems", lambda repo=Path("."): ["HEAD abc is behind origin/main (def); pull first"])

    assert ds.main([]) == 1
    err = capsys.readouterr().err
    assert "refusing to deploy" in err and "pull first" in err and "--allow-unsynced" in err
    assert calls == []                                     # refused before any Hub call

    assert ds.main(["--allow-unsynced"]) == 0
    [commit] = calls
    paths = {op.path_in_repo for op in commit["operations"]}
    assert {"app.py", "README.md", "requirements.txt"} <= paths and "old.py" in paths   # old.py: a delete operation
