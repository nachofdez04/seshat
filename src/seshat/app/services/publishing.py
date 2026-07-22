from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from seshat.app.services.job import JobNotFoundError
from seshat.core.config.settings import GitPublishingConfig
from seshat.core.models.documents import GeneratedDocument, document_is_publishable, effective_content
from seshat.core.models.enums import JobStatus
from seshat.core.models.publishing import PublishResult
from seshat.core.utils.log import get_logger
from seshat.core.utils.paths import safe_path_segment
from seshat.infra.git_publisher import git_ops, templates
from seshat.infra.git_publisher import pr as pr_module

if TYPE_CHECKING:
    from datetime import date
    from pathlib import Path

    from seshat.app.repositories.ops_repository import OpsRepository

logger = get_logger(__name__)


class PublishError(RuntimeError):
    """Publishing cannot proceed (feature disabled, bad job state, no approved documents)."""


class NothingToPublishError(PublishError):
    """Benign: the approved content already matches the target repository."""


class PublishGitError(PublishError):
    """A git or filesystem operation failed during the publish. Carries diagnostic detail."""


class PublishingService:
    def __init__(self, ops: OpsRepository, config: GitPublishingConfig | None = None) -> None:
        self._ops = ops
        self._config = config or GitPublishingConfig()
        # Single-flight: the target working tree is shared mutable state, so only one publish
        # may touch it at a time per process.
        self._publish_lock = asyncio.Lock()

    async def publish_job(self, job_id: str) -> PublishResult:
        cfg = self._config
        if not cfg.enabled:
            raise PublishError("Git publishing is disabled. Enable it with git_publishing.enabled=true")
        repo = cfg.target_repo_path
        if repo is None:
            raise PublishError("No target repository configured. Set git_publishing.target_repo_path")

        try:
            templates.validate_branch_name(cfg.base_branch)
        except ValueError as exc:
            raise PublishError(f"git_publishing.base_branch is not a valid branch name: {exc}") from exc

        row = await self._ops.get_job(job_id)
        if not row:
            raise JobNotFoundError(job_id)
        if row["status"] != JobStatus.DONE:
            raise PublishError("Documents can only be published for done jobs")

        doc_rows = await self._ops.get_documents_for_job(job_id)
        publishable = [doc for doc in map(GeneratedDocument.model_validate, doc_rows) if document_is_publishable(doc)]
        if not publishable:
            raise PublishError("No approved documents to publish")

        job_root, files = self._build_file_set(job_id, publishable)

        meeting_date: date | None = row["meeting_date"]
        try:
            branch = templates.build_branch_name(cfg.branch_prefix, job_id, meeting_date)
        except ValueError as exc:
            raise PublishError(f"Could not build a safe git branch name: {exc}") from exc

        commit_message = templates.build_commit_message(job_id, len(files))
        logger.info("Publishing %d doc(s) of job %s to %s", len(files), job_id, repo)

        async with self._publish_lock:
            commit_sha = await asyncio.to_thread(self._git_phase, repo, branch, job_root, files, commit_message)

        pr_title = templates.build_pr_title(job_id, len(files))
        pr_body = templates.build_pr_body(job_id, meeting_date, [rel_path for rel_path, _ in files])
        pr_url, compare_url = await asyncio.to_thread(self._pr_phase, repo, branch, pr_title, pr_body)

        result = PublishResult(
            job_id=job_id,
            branch=branch,
            commit_sha=commit_sha,
            pr_url=pr_url,
            compare_url=compare_url,
            files=[rel_path for rel_path, _ in files],
            published_at=datetime.now(UTC),
        )
        await self._ops.insert_publish_result(result)
        return result

    async def get_latest(self, job_id: str) -> PublishResult | None:
        row = await self._ops.get_latest_publish_result(job_id)
        return PublishResult.model_validate(row) if row else None

    def _build_file_set(self, job_id: str, documents: list[GeneratedDocument]) -> tuple[str, list[tuple[str, str]]]:
        """Build (job_root, [(repo-relative path, content)]) with every path segment validated."""
        try:
            job_segment = safe_path_segment(job_id, "job_id")
            files: list[tuple[str, str]] = []
            seen_filenames: set[str] = set()
            for doc in documents:
                filename = safe_path_segment(doc.filename, "filename")
                if not filename.casefold().endswith(".md"):
                    raise PublishError(f"Only Markdown documents can be published: {filename!r}")

                filename_key = filename.casefold()
                if filename_key in seen_filenames:
                    raise PublishError(f"Duplicate or ambiguous document filename: {filename!r}")

                seen_filenames.add(filename_key)
                kind = safe_path_segment(doc.kind.value, "kind")
                files.append((f"{self._config.docs_subdir}/{job_segment}/{kind}/{filename}", effective_content(doc)))
        except ValueError as exc:
            raise PublishError(str(exc)) from exc

        return f"{self._config.docs_subdir}/{job_segment}", files

    def _git_phase(
        self,
        repo: Path,
        branch: str,
        job_root: str,
        files: list[tuple[str, str]],
        commit_message: str,
    ) -> str:
        """Blocking git flow run under the publish lock inside a worker thread."""
        cfg = self._config
        try:
            git_ops.ensure_repo(repo, remote=cfg.target_remote)
            git_ops.ensure_clean(repo)
            git_ops.checkout_existing_branch(repo, cfg.base_branch)
            git_ops.pull(repo)
            git_ops.checkout_branch(repo, branch, base=cfg.base_branch)
            written = git_ops.sync_files(repo, job_root, files)
            try:
                commit_sha = git_ops.add_and_commit(repo, written, commit_message)
                logger.info("Created commit %s on %s", commit_sha, branch)
            except git_ops.EmptyCommitError as exc:
                head_sha = git_ops.get_head_sha(repo)
                base_sha = git_ops.get_branch_sha(repo, cfg.base_branch)
                remote_sha = git_ops.get_remote_branch_sha(repo, branch)
                if head_sha in (base_sha, remote_sha):
                    try:
                        git_ops.checkout_branch(repo, cfg.base_branch)
                    except git_ops.GitOperationError:
                        logger.warning("Could not return to the base branch after an empty commit")

                    raise NothingToPublishError(str(exc)) from exc

                # A previous attempt committed locally but failed at push; keep that commit
                # and resume the flow from the pending phase.
                commit_sha = git_ops.get_head_sha(repo, short=True)
                logger.info("Resuming publish of pending local commit %s", commit_sha)

            git_ops.push(repo, branch)
            logger.info("Push completed: %s", branch)
        except git_ops.GitOperationError as exc:
            raise PublishGitError(f"Git operation failed: {exc}") from exc

        return commit_sha

    def _pr_phase(self, repo: Path, branch: str, pr_title: str, pr_body: str) -> tuple[str, str]:
        """Blocking PR creation; on failure the publish stands and a compare URL is offered."""
        cfg = self._config
        try:
            existing_pr_url = pr_module.get_open_pr_url(
                repo_path=repo,
                branch=branch,
                base=cfg.base_branch,
                gh_executable=cfg.gh_executable,
            )
            if existing_pr_url:
                logger.info("Reusing open PR: %s", existing_pr_url)
                return existing_pr_url, ""

            pr_url = pr_module.create_pr(
                repo_path=repo,
                branch=branch,
                title=pr_title,
                body=pr_body,
                base=cfg.base_branch,
                gh_executable=cfg.gh_executable,
                draft=cfg.pr_draft,
            )
            logger.info("Created PR: %s", pr_url)
            return pr_url, ""
        except pr_module.PrCreationError as exc:
            logger.warning("Could not find or create the PR automatically: %s. The commit and push did complete.", exc)

        try:
            remote_url = git_ops.get_remote_url(repo)
        except git_ops.GitOperationError as exc:
            logger.warning("Could not read the remote URL for the compare fallback: %s", exc)
            remote_url = None

        compare_url = (pr_module.build_compare_url(remote_url, cfg.base_branch, branch) or "") if remote_url else ""
        if compare_url:
            logger.info("Open the PR manually at: %s", compare_url)

        return "", compare_url
