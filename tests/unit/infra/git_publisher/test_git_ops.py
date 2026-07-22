"""Tests for git_ops using real git repositories in tmp_path — subprocess is never mocked."""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from seshat.infra.git_publisher import git_ops
from seshat.infra.git_publisher.git_ops import (
    EmptyCommitError,
    GitOperationError,
    add_and_commit,
    checkout_branch,
    ensure_clean,
    ensure_repo,
    get_branch_sha,
    get_head_sha,
    get_remote_branch_sha,
    get_remote_url,
    is_clean,
    pull,
    sync_files,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not available in the environment")


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _init_repo(path: Path, initial_branch: str = "main") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-b", initial_branch)
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test User")
    (path / "README.md").write_text("# Test repo\n", encoding="utf-8")
    _git(path, "add", "README.md")
    _git(path, "commit", "-m", "Initial commit")
    return path


def _symlinks_supported(tmp_path: Path) -> bool:
    target = tmp_path / "symlink_target"
    target.mkdir(exist_ok=True)
    link = tmp_path / "symlink_probe"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        return False

    link.unlink()
    return True


class TestEnsureRepo:
    def test_valid_repo_passes(self, tmp_path: Path):
        repo = _init_repo(tmp_path / "repo")
        assert ensure_repo(repo) == repo

    def test_non_repo_raises(self, tmp_path: Path):
        (tmp_path / "notrepo").mkdir()
        with pytest.raises(GitOperationError, match="not a git repository"):
            ensure_repo(tmp_path / "notrepo")

    def test_missing_without_remote_raises(self, tmp_path: Path):
        with pytest.raises(GitOperationError, match="does not exist"):
            ensure_repo(tmp_path / "missing")

    def test_missing_with_remote_clones(self, tmp_path: Path):
        origin = tmp_path / "origin.git"
        _git(tmp_path, "init", "--bare", "-b", "main", str(origin))
        seed = _init_repo(tmp_path / "seed")
        _git(seed, "remote", "add", "origin", str(origin))
        _git(seed, "push", "-u", "origin", "main")

        clone = tmp_path / "clone"
        assert ensure_repo(clone, remote=str(origin)) == clone
        assert (clone / ".git").exists()
        assert (clone / "README.md").exists()

    def test_wraps_git_launch_error(self, tmp_path: Path):
        repo = _init_repo(tmp_path / "repo")
        with (
            patch("seshat.infra.git_publisher.git_ops.subprocess.run", side_effect=FileNotFoundError),
            pytest.raises(GitOperationError, match="Could not execute"),
        ):
            is_clean(repo)


class TestCheckoutBranch:
    def test_creates_new_branch_from_base(self, tmp_path: Path):
        repo = _init_repo(tmp_path / "repo")
        checkout_branch(repo, "feature/x", base="main")
        assert _git(repo, "branch", "--show-current") == "feature/x"

    def test_switches_to_existing_branch(self, tmp_path: Path):
        repo = _init_repo(tmp_path / "repo")
        _git(repo, "checkout", "-b", "existing")
        _git(repo, "checkout", "main")
        checkout_branch(repo, "existing")
        assert _git(repo, "branch", "--show-current") == "existing"

    def test_tracks_branch_that_only_exists_on_origin(self, tmp_path: Path):
        origin = tmp_path / "origin.git"
        _git(tmp_path, "init", "--bare", "-b", "main", str(origin))
        seed = _init_repo(tmp_path / "seed")
        _git(seed, "remote", "add", "origin", str(origin))
        _git(seed, "push", "-u", "origin", "main")
        _git(seed, "checkout", "-b", "develop")
        (seed / "develop.md").write_text("develop", encoding="utf-8")
        _git(seed, "add", "develop.md")
        _git(seed, "commit", "-m", "develop")
        _git(seed, "push", "-u", "origin", "develop")

        clone = tmp_path / "clone"
        _git(tmp_path, "clone", str(origin), str(clone))
        assert get_remote_branch_sha(clone, "develop") is not None
        checkout_branch(clone, "develop")
        assert _git(clone, "branch", "--show-current") == "develop"
        assert _git(clone, "rev-parse", "--abbrev-ref", "@{u}") == "origin/develop"

    def test_invalid_branch_name_rejected_before_touching_git(self, tmp_path: Path):
        repo = _init_repo(tmp_path / "repo")
        with pytest.raises(GitOperationError, match="Invalid branch name"):
            checkout_branch(repo, "bad..name")

    def test_missing_existing_branch_raises_without_creating_it(self, tmp_path: Path):
        repo = _init_repo(tmp_path / "repo")

        with pytest.raises(GitOperationError, match="does not exist"):
            git_ops.checkout_existing_branch(repo, "mian")

        assert get_branch_sha(repo, "mian") is None


class TestPull:
    def test_no_upstream_is_not_an_error(self, tmp_path: Path):
        repo = _init_repo(tmp_path / "repo")
        pull(repo)  # must not raise


class TestSyncFiles:
    @pytest.mark.parametrize(
        "rel_path",
        [
            "../escape.md",
            "/absolute/path.md",
            "C:\\absolute\\path.md",
            "docs/../../escape.md",
            ".git/config",
            "docs/.git/hook.md",
            "docs/CON/file.md".replace("CON", "bad\x1fname"),
            "docs/que?stion.md",
            "docs/sta*r.md",
        ],
    )
    def test_blocks_unsafe_paths(self, tmp_path: Path, rel_path: str):
        repo = _init_repo(tmp_path / "repo")
        with pytest.raises(GitOperationError, match="blocked"):
            sync_files(repo, "docs", [(rel_path, "x")])

        assert not (tmp_path / "escape.md").exists()

    def test_prevalidates_every_path_before_writing(self, tmp_path: Path):
        repo = _init_repo(tmp_path / "repo")
        with pytest.raises(GitOperationError, match="path traversal blocked"):
            sync_files(repo, "docs", [("docs/safe.md", "safe"), ("../escape.md", "escape")])

        assert not (repo / "docs" / "safe.md").exists()

    def test_confines_writes_to_allowed_root(self, tmp_path: Path):
        repo = _init_repo(tmp_path / "repo")
        with pytest.raises(GitOperationError, match="authorized subtree"):
            sync_files(repo, "docs/meetings/m1", [("README.md", "corrupt")])

        assert (repo / "README.md").read_text(encoding="utf-8") == "# Test repo\n"

    def test_rejects_duplicate_paths(self, tmp_path: Path):
        repo = _init_repo(tmp_path / "repo")
        with pytest.raises(GitOperationError, match="Duplicate publish path"):
            sync_files(repo, "docs", [("docs/a.md", "1"), ("docs/a.md", "2")])

    def test_writes_files_and_creates_subdirs(self, tmp_path: Path):
        repo = _init_repo(tmp_path / "repo")
        written = sync_files(repo, "docs/m1", [("docs/m1/kind/a.md", "# A")])
        assert (repo / "docs" / "m1" / "kind" / "a.md").read_text(encoding="utf-8") == "# A"
        assert len(written) == 1

    def test_removes_files_not_in_desired_set_and_prunes_empty_dirs(self, tmp_path: Path):
        repo = _init_repo(tmp_path / "repo")
        root = "docs/m1"
        sync_files(repo, root, [(f"{root}/adr/a.md", "A old"), (f"{root}/adr/b.md", "B old")])

        affected = sync_files(repo, root, [(f"{root}/summary/a.md", "A new")])

        assert (repo / root / "summary" / "a.md").read_text(encoding="utf-8") == "A new"
        assert not (repo / root / "adr").exists()
        assert any(path.name == "b.md" for path in affected)

    def test_blocks_git_dir_inside_subtree(self, tmp_path: Path):
        repo = _init_repo(tmp_path / "repo")
        nested_git = repo / "docs" / "m1" / ".git"
        nested_git.mkdir(parents=True)
        with pytest.raises(GitOperationError, match=r"\.git"):
            sync_files(repo, "docs/m1", [("docs/m1/a.md", "A")])

    def test_blocks_symlink_inside_subtree(self, tmp_path: Path):
        if not _symlinks_supported(tmp_path):
            pytest.skip("symlinks not supported in this environment")

        repo = _init_repo(tmp_path / "repo")
        outside = tmp_path / "outside"
        outside.mkdir()
        subtree = repo / "docs" / "m1"
        subtree.mkdir(parents=True)
        (subtree / "link").symlink_to(outside, target_is_directory=True)
        with pytest.raises(GitOperationError, match="symlink"):
            sync_files(repo, "docs/m1", [("docs/m1/a.md", "A")])

    def test_blocks_symlinked_ancestor(self, tmp_path: Path):
        if not _symlinks_supported(tmp_path):
            pytest.skip("symlinks not supported in this environment")

        repo = _init_repo(tmp_path / "repo")
        outside = tmp_path / "outside"
        outside.mkdir()
        (repo / "docs").symlink_to(outside, target_is_directory=True)
        with pytest.raises(GitOperationError, match="symlink"):
            sync_files(repo, "docs/m1", [("docs/m1/a.md", "A")])


class TestAddAndCommit:
    def test_creates_commit_and_returns_short_sha(self, tmp_path: Path):
        repo = _init_repo(tmp_path / "repo")
        written = sync_files(repo, "docs", [("docs/new.md", "# New")])
        sha = add_and_commit(repo, written, "docs: add new.md")
        assert 7 <= len(sha) <= 12
        assert "docs: add new.md" in _git(repo, "log", "--oneline", "-1")

    def test_no_changes_raises_empty_commit_error(self, tmp_path: Path):
        repo = _init_repo(tmp_path / "repo")
        written = sync_files(repo, "docs", [("docs/dup.md", "same")])
        add_and_commit(repo, written, "first")
        written_again = sync_files(repo, "docs", [("docs/dup.md", "same")])
        with pytest.raises(EmptyCommitError, match="Nothing to publish"):
            add_and_commit(repo, written_again, "second")

    def test_stages_deletions_from_sync(self, tmp_path: Path):
        repo = _init_repo(tmp_path / "repo")
        root = "docs/m1"
        initial = sync_files(repo, root, [(f"{root}/a.md", "A"), (f"{root}/b.md", "B")])
        add_and_commit(repo, initial, "initial docs")

        affected = sync_files(repo, root, [(f"{root}/a.md", "A changed")])
        add_and_commit(repo, affected, "sync docs")

        status = _git(repo, "show", "--format=", "--name-status", "HEAD")
        assert f"D\t{root}/b.md" in status.replace("\\", "/")

    def test_path_outside_repo_rejected(self, tmp_path: Path):
        repo = _init_repo(tmp_path / "repo")
        outside = tmp_path / "outside.md"
        outside.write_text("x", encoding="utf-8")
        with pytest.raises(GitOperationError, match="outside the repository"):
            add_and_commit(repo, [outside], "bad")


class TestCleanliness:
    def test_clean_repo_passes(self, tmp_path: Path):
        repo = _init_repo(tmp_path / "repo")
        assert is_clean(repo) is True
        ensure_clean(repo)  # must not raise

    def test_dirty_repo_detected(self, tmp_path: Path):
        repo = _init_repo(tmp_path / "repo")
        (repo / "untracked.md").write_text("uncommitted changes", encoding="utf-8")
        assert is_clean(repo) is False
        with pytest.raises(GitOperationError, match="uncommitted changes"):
            ensure_clean(repo)


class TestShaHelpers:
    def test_head_and_branch_shas_agree(self, tmp_path: Path):
        repo = _init_repo(tmp_path / "repo")
        assert get_head_sha(repo) == get_branch_sha(repo, "main")
        assert get_branch_sha(repo, "missing") is None
        assert get_remote_branch_sha(repo, "main") is None

    def test_remote_url_absent_returns_none(self, tmp_path: Path):
        repo = _init_repo(tmp_path / "repo")
        assert get_remote_url(repo) is None
        _git(repo, "remote", "add", "origin", "https://github.com/acme/docs.git")
        assert get_remote_url(repo) == "https://github.com/acme/docs.git"
