import os

import httpx

_BEDROCK_PROFILE = os.environ.get("AWS_PROFILE") or "ClaudeCode"


def _bedrock_available(profile_name: str | None = None) -> bool:
    try:
        import boto3

        return boto3.Session(profile_name=profile_name).get_credentials() is not None
    except Exception:
        return False


def _azure_available() -> bool:
    return bool(
        os.environ.get("AZURE_OPENAI_ENDPOINT")
        and os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        and os.environ.get("AZURE_OPENAI_API_KEY")
    )


def _openai_reachable(openai_api_key_env_var: str | None = None) -> bool:
    if _azure_available():
        return True

    key = os.environ.get(openai_api_key_env_var or "OPENAI_API_KEY")
    if not key:
        return False

    try:
        response = httpx.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=5,
        )
        return response.status_code < 400
    except httpx.RequestError:
        return False


def _anthropic_reachable() -> bool:
    if _bedrock_available(profile_name=_BEDROCK_PROFILE):
        return True

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return False

    try:
        response = httpx.get(
            "https://api.anthropic.com/v1/models",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            timeout=5,
        )
        return response.status_code < 400
    except httpx.RequestError:
        return False


def _cohere_reachable() -> bool:
    key = os.environ.get("COHERE_API_KEY")
    if not key:
        return False

    try:
        response = httpx.get(
            "https://api.cohere.com/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=5,
        )
        return response.status_code < 400
    except httpx.RequestError:
        return False


def _voyage_reachable() -> bool:
    key = os.environ.get("VOYAGE_API_KEY")
    if not key:
        return False

    try:
        # Voyage has no lightweight reachability endpoint (no /models); use a minimal rerank call.
        response = httpx.post(
            "https://api.voyageai.com/v1/rerank",
            headers={"Authorization": f"Bearer {key}"},
            json={"query": "ping", "documents": ["ping"], "model": "rerank-2"},
            timeout=5,
        )
        return response.status_code < 400
    except httpx.RequestError:
        return False


def _openai_direct_reachable() -> bool:
    """Check OPENAI_API_KEY and api.openai.com reachability — no Azure fallback."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return False

    try:
        response = httpx.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=5,
        )
        return response.status_code < 400
    except httpx.RequestError:
        return False


def _assemblyai_reachable() -> bool:
    key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not key:
        return False

    try:
        response = httpx.get(
            "https://api.assemblyai.com/v2",
            headers={"Authorization": key},
            timeout=5,
        )
        return response.status_code < 400
    except httpx.RequestError:
        return False
