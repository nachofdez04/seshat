"""Synchronous wrappers over the git CLI for operations on the target docs repo.

All functions are blocking; the service layer offloads them with ``asyncio.to_thread``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path, PurePosixPath, PureWindowsPath

from seshat.infra.git_publisher.templates import validate_branch_name

# Force English git messages so text-based checks (e.g. "no tracking information") do not
# depend on the locale configured on the host machine.
_GIT_ENV = {**os.environ, "LC_ALL": "C", "LANG": "C"}


class GitOperationError(RuntimeError):
    """A git operation failed. Carries stdout/stderr for diagnosis."""

    def __init__(self, message: str, stdout: str = "", stderr: str = "") -> None:
        detail = "\n".join(filter(None, [message, stdout.strip(), stderr.strip()]))
        super().__init__(detail)
        self.stdout = stdout
        self.stderr = stderr


class EmptyCommitError(GitOperationError):
    """Nothing to commit: the approved content already matches the repository.

    Benign subtype of `GitOperationError` so upper layers can distinguish "already up to date"
    from a real git failure and surface it as information, not an error.
    """


def _run(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            env=_GIT_ENV,
        )
    except OSError as exc:
        raise GitOperationError(f"Could not execute {args[0]!r}: {exc}") from exc

    if check and result.returncode != 0:
        raise GitOperationError(
            f"git {args[1] if len(args) > 1 else ''!r} failed (exit code {result.returncode})",
            stdout=result.stdout,
            stderr=result.stderr,
        )

    return result


def ensure_repo(target_path: Path, remote: str | None = None) -> Path:
    """Ensure target_path is a valid git repo.

    - Missing and remote given: clone.
    - Exists with .git: fetch (no-op when no remote is configured).
    - Exists without .git: raise GitOperationError.
    """
    git_dir = target_path / ".git"
    if not target_path.exists():
        if not remote:
            raise GitOperationError(f"Target directory does not exist and no remote was given: {target_path}")

        target_path.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", remote, str(target_path)], cwd=target_path.parent)
    elif not git_dir.exists():
        raise GitOperationError(f"{target_path} exists but is not a git repository (missing .git/)")
    elif remote or get_remote_url(target_path):
        _run(["git", "fetch", "--all", "--prune"], cwd=target_path)

    return target_path


def _ref_sha(repo: Path, ref: str) -> str | None:
    result = _run(["git", "rev-parse", "--verify", ref], cwd=repo, check=False)
    if result.returncode != 0:
        return None

    sha = result.stdout.strip()
    return sha or None


def get_head_sha(repo: Path, *, short: bool = False) -> str:
    """Return the SHA of HEAD, optionally in its short form."""
    flag = "--short" if short else "--verify"
    result = _run(["git", "rev-parse", flag, "HEAD"], cwd=repo)
    return result.stdout.strip()


def get_branch_sha(repo: Path, branch: str) -> str | None:
    """Return the SHA of a local branch, or ``None`` if it does not exist."""
    return _ref_sha(repo, f"refs/heads/{branch}")


def get_remote_branch_sha(repo: Path, branch: str, remote: str = "origin") -> str | None:
    """Return the last known SHA of ``remote/branch``, or ``None`` if it does not exist."""
    return _ref_sha(repo, f"refs/remotes/{remote}/{branch}")


def get_remote_url(repo: Path, remote: str = "origin") -> str | None:
    """Return the URL of the given remote, or None when it is not configured."""
    result = _run(["git", "remote", "get-url", remote], cwd=repo, check=False)
    url = result.stdout.strip()
    return url or None


def is_clean(repo: Path) -> bool:
    """True when the working tree and index have no uncommitted changes."""
    result = _run(["git", "status", "--porcelain"], cwd=repo)
    return result.stdout.strip() == ""


def ensure_clean(repo: Path) -> None:
    """Raise GitOperationError when the target repo has uncommitted changes.

    Prevents mixing manual work in the target repo with the auto-generated documents.
    """
    if not is_clean(repo):
        raise GitOperationError(
            "The target repository has uncommitted changes; resolve them (commit or stash) before publishing."
        )


def checkout_branch(repo: Path, branch: str, base: str | None = None) -> None:
    """Check out a local branch, track its remote counterpart, or create it from ``base``."""
    try:
        validate_branch_name(branch)
        if base is not None:
            validate_branch_name(base)
    except ValueError as exc:
        raise GitOperationError(f"Invalid branch name: {exc}") from exc

    if get_branch_sha(repo, branch) is not None:
        _run(["git", "checkout", branch], cwd=repo)
    elif get_remote_branch_sha(repo, branch) is not None:
        _run(["git", "checkout", "--track", "-b", branch, f"origin/{branch}"], cwd=repo)
    elif base:
        _run(["git", "checkout", "-b", branch, base], cwd=repo)
    else:
        _run(["git", "checkout", "-b", branch], cwd=repo)


def checkout_existing_branch(repo: Path, branch: str) -> None:
    """Check out an existing local branch or track its remote counterpart."""
    try:
        validate_branch_name(branch)
    except ValueError as exc:
        raise GitOperationError(f"Invalid branch name: {exc}") from exc

    if get_branch_sha(repo, branch) is not None:
        _run(["git", "checkout", branch], cwd=repo)
    elif get_remote_branch_sha(repo, branch) is not None:
        _run(["git", "checkout", "--track", "-b", branch, f"origin/{branch}"], cwd=repo)
    else:
        raise GitOperationError(f"Branch does not exist locally or on origin: {branch}")


def pull(repo: Path) -> None:
    """Run git pull --ff-only on the current branch."""
    result = _run(["git", "pull", "--ff-only"], cwd=repo, check=False)
    # Not an error when no upstream is configured (freshly cloned repo with a single branch).
    if result.returncode != 0 and "no tracking information" not in result.stderr:
        raise GitOperationError(
            "git pull failed",
            stdout=result.stdout,
            stderr=result.stderr,
        )


_INVALID_PORTABLE_PATH_CHARS = set('<>:"|?*')


def _relative_parts(rel_path: str | Path) -> tuple[str, ...]:
    text = str(rel_path)
    portable = PurePosixPath(text.replace("\\", "/"))
    windows = PureWindowsPath(text)
    if not text or "\x00" in text or portable.is_absolute() or windows.is_absolute() or bool(windows.drive):
        raise GitOperationError(f"Path not allowed (path traversal blocked): {rel_path}")

    parts = portable.parts
    if not parts:
        raise GitOperationError(f"Path not allowed (path traversal blocked): {rel_path}")

    for part in parts:
        if (
            part in {"", ".", ".."}
            or part.casefold() == ".git"
            or any(ord(char) < 32 or char in _INVALID_PORTABLE_PATH_CHARS for char in part)
        ):
            raise GitOperationError(f"Path not allowed (path traversal blocked): {rel_path}")

    return parts


def _is_link_like(path: Path) -> bool:
    junction_check = getattr(path, "is_junction", None)
    return path.is_symlink() or bool(junction_check and junction_check())


def _resolve_repo_path(repo_resolved: Path, rel_path: str | Path) -> Path:
    parts = _relative_parts(rel_path)
    current = repo_resolved
    for part in parts:
        current = current / part
        if _is_link_like(current):
            raise GitOperationError(f"Path not allowed (symlink blocked): {rel_path}")

    target = current.resolve()
    if not target.is_relative_to(repo_resolved):
        raise GitOperationError(f"Path outside the repository (path traversal blocked): {rel_path}")

    return target


def _prepare_targets(
    repo: Path,
    files: list[tuple[str, str]],
    *,
    allowed_root: str | Path | None = None,
) -> tuple[Path, list[tuple[Path, str]]]:
    repo_resolved = repo.resolve()
    root = _resolve_repo_path(repo_resolved, allowed_root) if allowed_root is not None else None
    prepared: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for rel_path, content in files:
        target = _resolve_repo_path(repo_resolved, rel_path)
        if root is not None and (target == root or not target.is_relative_to(root)):
            raise GitOperationError(f"Path outside the authorized subtree (path traversal blocked): {rel_path}")

        if target in seen:
            raise GitOperationError(f"Duplicate publish path: {rel_path}")

        seen.add(target)
        prepared.append((target, content))

    return repo_resolved, prepared


def sync_files(repo: Path, root: str | Path, files: list[tuple[str, str]]) -> list[Path]:
    """Synchronize ``root`` with ``files``; return the paths to stage, deletions included.

    Every path is pre-validated before anything is written or deleted. The subtree belongs
    exclusively to one job, so files that no longer appear in ``files`` are removed.
    """
    repo_resolved, prepared = _prepare_targets(repo, files, allowed_root=root)
    root_path = _resolve_repo_path(repo_resolved, root)
    if root_path.exists() and not root_path.is_dir():
        raise GitOperationError(f"The publish subtree is not a directory: {root}")

    try:
        entries = list(root_path.rglob("*")) if root_path.exists() else []
    except OSError as exc:
        raise GitOperationError(f"Could not inspect the publish subtree: {exc}") from exc

    if any(entry.name.casefold() == ".git" for entry in entries):
        raise GitOperationError("Blocked a .git directory inside the publish subtree")

    for entry in entries:
        if _is_link_like(entry):
            raise GitOperationError(f"Blocked a symlink inside the publish subtree: {entry}")

        try:
            entry_resolved = entry.resolve()
        except (OSError, RuntimeError) as exc:
            raise GitOperationError(f"Could not validate the existing path {entry}: {exc}") from exc

        if not entry_resolved.is_relative_to(root_path):
            raise GitOperationError(f"Existing path outside the authorized subtree: {entry}")

    desired = {target for target, _ in prepared}
    stale = [entry for entry in entries if entry.is_file() and entry.resolve() not in desired]

    try:
        for target, content in prepared:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

        for path in stale:
            path.unlink(missing_ok=True)

        for directory in sorted(
            (entry for entry in entries if entry.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            if directory.exists() and not any(directory.iterdir()):
                directory.rmdir()
    except OSError as exc:
        raise GitOperationError(f"Could not synchronize the publish subtree: {exc}") from exc

    return [target for target, _ in prepared] + stale


def add_and_commit(repo: Path, paths: list[Path], message: str) -> str:
    """Run git add on the paths and create the commit. Returns the short SHA.

    When nothing is staged after the `git add` (the approved content already matches the repo),
    no commit is attempted and `EmptyCommitError` is raised instead of git's confusing message.
    """
    repo_resolved = repo.resolve()
    str_paths: list[str] = []
    seen: set[Path] = set()
    for path in paths:
        candidate = path.resolve() if path.is_absolute() else (repo_resolved / path).resolve()
        if not candidate.is_relative_to(repo_resolved):
            raise GitOperationError(f"Path to stage is outside the repository: {path}")

        relative = candidate.relative_to(repo_resolved)
        _relative_parts(relative)
        if candidate not in seen:
            seen.add(candidate)
            str_paths.append(str(relative))

    if str_paths:
        _run(["git", "add", "-A", "--", *str_paths], cwd=repo)

    staged = _run(["git", "diff", "--cached", "--quiet"], cwd=repo, check=False)
    if staged.returncode == 0:
        raise EmptyCommitError("Nothing to publish: the approved content already matches the repository.")
    if staged.returncode != 1:
        raise GitOperationError(
            "Could not inspect the staged content",
            stdout=staged.stdout,
            stderr=staged.stderr,
        )

    _run(["git", "commit", "-m", message], cwd=repo)
    result = _run(["git", "rev-parse", "--short", "HEAD"], cwd=repo)
    return result.stdout.strip()


def push(repo: Path, branch: str) -> None:
    """Run git push -u origin <branch>."""
    _run(["git", "push", "-u", "origin", branch], cwd=repo)
