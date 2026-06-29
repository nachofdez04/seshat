from __future__ import annotations

from pydantic import BaseModel

from seshat.models.nodes import KBNode


class NodeListResponse(BaseModel):
    nodes: list[KBNode]


class NodeDetailResponse(BaseModel):
    node: KBNode
    neighbours: list[KBNode]


class ImpactNode(BaseModel):
    node: KBNode
    traversal_depth: int


class ImpactResponse(BaseModel):
    nodes: list[ImpactNode]


class JobSubmitResponse(BaseModel):
    job_id: str


class JobActionResponse(BaseModel):
    status: str
