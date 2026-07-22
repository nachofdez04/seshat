"""Wrapper over the gh CLI to create pull requests, with a manual-compare fallback."""

from __future__ import annotations

import re
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

# Extracts owner/repo from a GitHub remote URL (https or ssh).
_GITHUB_REMOTE_RE = re.compile(r"github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$")
_GH_COMMAND_TIMEOUT_S = 60.0


class PrCreationError(RuntimeError):
    """Creating the PR with the gh CLI failed."""

    def __init__(self, message: str, stdout: str = "", stderr: str = "") -> None:
        detail = "\n".join(filter(None, [message, stdout.strip(), stderr.strip()]))
        super().__init__(detail)
        self.stdout = stdout
        self.stderr = stderr


def get_open_pr_url(
    repo_path: Path,
    branch: str,
    base: str = "main",
    gh_executable: str = "gh",
    timeout_s: float = _GH_COMMAND_TIMEOUT_S,
) -> str | None:
    """Return the URL of the open PR for branch into base, or None when absent."""
    cmd = [
        gh_executable,
        "pr",
        "list",
        "--head",
        branch,
        "--base",
        base,
        "--state",
        "open",
        "--json",
        "url",
        "--jq",
        '.[0].url // ""',
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise PrCreationError(f"gh pr list timed out ({timeout_s:g}s)") from exc
    except OSError as exc:
        raise PrCreationError(f"Could not execute {gh_executable!r}: {exc}") from exc

    if result.returncode != 0:
        raise PrCreationError(
            f"gh pr list failed (exit code {result.returncode})",
            stdout=result.stdout,
            stderr=result.stderr,
        )

    url = result.stdout.strip()
    if not url:
        return None
    if not url.startswith("http"):
        raise PrCreationError(
            "gh pr list did not return a valid URL",
            stdout=result.stdout,
            stderr=result.stderr,
        )

    return url


def create_pr(
    repo_path: Path,
    branch: str,
    title: str,
    body: str,
    base: str = "main",
    gh_executable: str = "gh",
    draft: bool = False,
    timeout_s: float = _GH_COMMAND_TIMEOUT_S,
) -> str:
    """Create a PR with the gh CLI and return the URL of the created PR.

    Raises PrCreationError when gh is unavailable or the command fails.
    """
    cmd = [
        gh_executable,
        "pr",
        "create",
        "--title",
        title,
        "--body",
        body,
        "--base",
        base,
        "--head",
        branch,
    ]
    if draft:
        cmd.append("--draft")

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise PrCreationError(f"gh pr create timed out ({timeout_s:g}s)") from exc
    except OSError as exc:
        raise PrCreationError(f"Could not execute {gh_executable!r}: {exc}") from exc

    if result.returncode != 0:
        raise PrCreationError(
            f"gh pr create failed (exit code {result.returncode})",
            stdout=result.stdout,
            stderr=result.stderr,
        )

    # gh pr create prints the URL on the last stdout line. When stdout is empty (gh returned 0
    # without writing anything), avoid the IndexError and raise a clear error instead.
    lines = result.stdout.strip().splitlines()
    url = lines[-1].strip() if lines else ""
    if not url.startswith("http"):
        raise PrCreationError(
            "gh pr create did not return a valid URL",
            stdout=result.stdout,
            stderr=result.stderr,
        )

    return url


def build_compare_url(remote_url: str, base: str, branch: str) -> str | None:
    """Build the GitHub 'compare' URL to open the PR manually.

    Returns None when the remote is not GitHub. Useful as a fallback when `gh` fails or is absent.
    """
    match = _GITHUB_REMOTE_RE.search(remote_url)
    if not match:
        return None

    owner = match.group("owner")
    repo = match.group("repo")
    return f"https://github.com/{owner}/{repo}/compare/{base}...{branch}?expand=1"
