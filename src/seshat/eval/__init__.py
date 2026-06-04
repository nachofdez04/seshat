from seshat.eval.identification.runner import IdentificationEvalRunner
from seshat.eval.models import GateResult
from seshat.eval.resolution.runner import ResolutionEvalRunner
from seshat.eval.retrieval.runner import RetrievalEvalRunner


def require_eval_deps() -> None:
    try:
        import rapidfuzz  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The seshat.eval package requires optional dependencies that are not installed. Run: `uv sync --group eval`"
        ) from exc


__all__ = [
    "GateResult",
    "IdentificationEvalRunner",
    "ResolutionEvalRunner",
    "RetrievalEvalRunner",
    "require_eval_deps",
]
