import logging
import re

logger = logging.getLogger(__name__)


def ensure_psycopg_scheme(url: str, warn_msg: str | None = None) -> str:
    """Normalize a PostgreSQL URL to use the psycopg (v3) driver scheme."""
    match = re.match(r"^postgresql(\+\w+)?://", url)
    if match is None:
        raise ValueError("Invalid connection string: must start with 'postgresql://' or 'postgresql+<driver>://'")

    qualifier = match.group(1)
    if qualifier == "+psycopg":
        return url

    if qualifier is not None:
        warn_msg = warn_msg or "Unexpected driver %r in connection string; replacing with '+psycopg'"
        logger.warning(warn_msg, qualifier)

    return re.sub(r"^postgresql(\+\w+)?://", "postgresql+psycopg://", url)
