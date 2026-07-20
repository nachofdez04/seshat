from __future__ import annotations

import json
from typing import Any

import httpx


class ApiClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"X-API-Key": api_key}

    def _get(self, path: str, **params: Any) -> Any:
        response = self._request("GET", path, params=params)
        response.raise_for_status()
        return response.json()

    def _get_safe(self, path: str, **params: Any) -> Any | None:
        try:
            return self._get(path, **params)
        except httpx.HTTPStatusError:
            return None

    def _post(self, path: str, json_body: Any = None, **kwargs: Any) -> httpx.Response:
        return self._request("POST", path, json=json_body, **kwargs)

    def _put(self, path: str, json_body: Any) -> httpx.Response:
        return self._request("PUT", path, json=json_body)

    def _delete(self, path: str, params: dict | None = None) -> httpx.Response:
        return self._request("DELETE", path, params=params)

    def _request(self, method: str, path: str, *, timeout: int = 30, **kwargs: Any) -> httpx.Response:
        url = f"{self._base_url}{path}"
        return httpx.request(method, url, headers=self._headers, timeout=timeout, **kwargs)

    # -- Health / Identity -----------------------------------------------------

    def get_health(self) -> dict:
        return self._get("/v1/health/components")

    def whoami(self) -> dict | None:
        return self._get_safe("/v1/me")

    def is_root_api_key_valid(self) -> bool:
        return self._get_safe("/v1/admin/api-keys") is not None

    # -- Jobs ------------------------------------------------------------------

    def submit_job(self, file_bytes: bytes, filename: str, body: dict) -> httpx.Response:
        return self._post(
            "/v1/jobs",
            files={"file": (filename, file_bytes, "application/octet-stream")},
            data={"body": json.dumps(body)},
            timeout=60,
        )

    def list_jobs(
        self,
        status: str | None = None,
        source_type: str | None = None,
        meeting_date_from: str | None = None,
        meeting_date_to: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        params: dict = {"limit": limit, "offset": offset}
        if status:
            params["job_status"] = status
        if source_type:
            params["source_type"] = source_type
        if meeting_date_from:
            params["meeting_date_from"] = meeting_date_from
        if meeting_date_to:
            params["meeting_date_to"] = meeting_date_to
        return self._get("/v1/jobs", **params)

    def get_job(self, job_id: str) -> dict:
        return self._get(f"/v1/jobs/{job_id}")

    def get_job_results(self, job_id: str) -> dict:
        return self._get(f"/v1/jobs/{job_id}/results")

    def approve_job(self, job_id: str, payload: dict) -> httpx.Response:
        return self._post(f"/v1/jobs/{job_id}/approve", json_body=payload)

    def retry_job(self, job_id: str) -> httpx.Response:
        return self._post(f"/v1/jobs/{job_id}/retry")

    def get_transcript_excerpt(self, job_id: str, char_start: int, char_end: int) -> str | None:
        resp = self._get_safe(f"/v1/jobs/{job_id}/transcript/excerpt", char_start=char_start, char_end=char_end)
        return resp.get("text") if resp else None

    # -- Graph -----------------------------------------------------------------

    def query_graph(self, **filters: Any) -> list[dict]:
        resp = self._get("/v1/graph", **filters)
        return resp.get("nodes", [])

    def search_graph(self, q: str, limit: int = 10, search_mode: str = "semantic", **filters: Any) -> list[dict]:
        resp = self._get("/v1/graph/search", q=q, limit=limit, search_mode=search_mode, **filters)
        return resp.get("results", [])

    def get_node(self, node_id: str) -> dict:
        return self._get(f"/v1/graph/{node_id}")

    def get_node_detail(self, node_id: str) -> dict:
        return self._get(f"/v1/graph/{node_id}/detail")

    def get_node_impact(
        self,
        node_id: str,
        depth: int = 2,
        rel_types: str | None = None,
        min_confidence: float = 0.0,
        direction: str = "outbound",
    ) -> dict:
        params: dict = {"depth": depth, "direction": direction}
        if rel_types:
            params["rel_types"] = rel_types
        if min_confidence > 0.0:
            params["min_confidence"] = min_confidence
        resp = self._get_safe(f"/v1/graph/{node_id}/impact", **params)
        return resp if resp else {"nodes": [], "relationships": []}

    def create_node(self, payload: dict) -> httpx.Response:
        return self._post("/v1/graph/nodes", json_body=payload)

    def update_node(self, node_id: str, payload: dict) -> httpx.Response:
        return self._put(f"/v1/graph/nodes/{node_id}", json_body=payload)

    def override_node(self, node_id: str, payload: dict) -> httpx.Response:
        return self._put(f"/v1/graph/nodes/{node_id}/override", json_body=payload)

    def delete_node(self, node_id: str, *, cascade: bool = True) -> httpx.Response:
        return self._delete(f"/v1/graph/nodes/{node_id}", params={"cascade": cascade})

    def resolve_nodes(self, node_ids: list[str]) -> httpx.Response:
        return self._post("/v1/graph/nodes/resolve", json_body={"node_ids": node_ids})

    def list_relationships(
        self,
        node_id: str | None = None,
        rel_type: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        params: dict = {"limit": limit}
        if node_id:
            params["node_id"] = node_id
        if rel_type:
            params["rel_type"] = rel_type
        resp = self._get("/v1/graph/relationships", **params)
        return resp.get("relationships", [])

    def create_relationship(self, source_id: str, target_id: str, rel_type: str) -> httpx.Response:
        json_body = {"source_id": source_id, "target_id": target_id, "rel_type": rel_type}
        return self._post("/v1/graph/relationships", json_body=json_body)

    def delete_relationship(self, rel_id: str) -> httpx.Response:
        return self._delete(f"/v1/graph/relationships/{rel_id}")

    # -- Admin (root key) ------------------------------------------------------

    def list_api_keys(self) -> list[dict]:
        return self._get("/v1/admin/api-keys")

    def create_api_key(self, user_id: str, role: str) -> httpx.Response:
        return self._post("/v1/admin/api-keys", json_body={"user_id": user_id, "role": role})

    def revoke_api_key(self, key_id: int) -> httpx.Response:
        return self._delete(f"/v1/admin/api-keys/{key_id}")
