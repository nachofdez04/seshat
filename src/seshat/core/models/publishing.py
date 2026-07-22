from datetime import datetime

from pydantic import Field

from seshat.core.models.base import SeshatModel


class PublishResult(SeshatModel):
    job_id: str
    branch: str
    commit_sha: str
    pr_url: str = Field(default="", description="URL of the created PR; empty when gh failed or is absent.")
    compare_url: str = Field(
        default="", description="GitHub 'compare' URL to open the PR manually when pr_url is empty."
    )
    files: list[str] = Field(description="Repo-relative paths published to the target repository.")
    published_at: datetime = Field(description="UTC timestamp of the publish.")
