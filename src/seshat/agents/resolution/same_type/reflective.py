from __future__ import annotations

import json
from typing import TYPE_CHECKING, Generic, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from seshat.agents.base import RetryExhaustedError
from seshat.agents.resolution.base import (
    ResolvedRelationship,
    _BaseResolutionAgent,
    _ResultBase,
    _SameTypeEntry,
)
from seshat.models.enums import RelationshipType
from seshat.utils.log import get_logger

if TYPE_CHECKING:
    from uuid import UUID

    from langchain_core.language_models import BaseChatModel

    from seshat.models.nodes import KBNode

logger = get_logger(__name__)

E = TypeVar("E", bound=_SameTypeEntry)


_TIEBREAKER_PROMPT = """\
Two relationship types are competing for each item below.
Using the definitions and selection rules in the system prompt above, choose the better fit.
Return exactly one decision per item, in the same order as the input list.

<contested_relationships>
{relations_json}
</contested_relationships>
"""


class _SelfReviewRetryExhaustedError(RetryExhaustedError):
    pass


class TiebreakerEntry(BaseModel):
    chosen: str = Field(description="The winning rel_type value.")
    rationale: str = Field(description="One sentence explaining the choice.")


class TiebreakerResult(BaseModel):
    decisions: list[TiebreakerEntry] = Field(
        description="One decision per contested relationship, in the same order as the input list.",
    )


class ReflectiveResolutionAgent(_BaseResolutionAgent[E], Generic[E]):
    """Wraps any _BaseResolutionAgent with a competing-hypothesis tiebreaker for same-type entries.

    After the inner agent resolves relationships for a source node, entries where the extractor
    flagged a competing alt_rel_type are sent to a single tiebreaker LLM call that chooses
    between the two options. Entries without alt_rel_type bypass the tiebreaker entirely.
    Degrades gracefully to the extractor's original rel_type on any tiebreaker failure.
    """

    def __init__(
        self,
        inner: _BaseResolutionAgent[E],
        review_llm: BaseChatModel,
    ) -> None:
        super().__init__(llm=inner._llm, config=inner._config)
        self._inner = inner
        self._review_llm = review_llm

    @property
    def name(self) -> str:
        return f"Reflective{self._inner.name}"

    @property
    def _result_model(self) -> type[_ResultBase[E]]:
        return self._inner._result_model

    @property
    def _system_prompt(self) -> str:
        return self._inner._system_prompt

    def _validate_relationships(
        self,
        relationships: list[ResolvedRelationship],
    ) -> tuple[list[ResolvedRelationship], list[ResolvedRelationship]]:
        return self._inner._validate_relationships(relationships)

    def prompt_texts(self) -> dict[str, str]:
        return self._inner.prompt_texts() | {"tiebreaker": _TIEBREAKER_PROMPT}

    async def _run_for_source(
        self,
        source: KBNode,
        targets: list[KBNode],
        siblings: list[KBNode] | None = None,
    ) -> tuple[list[E], dict[str, UUID]]:
        entries, id_map = await self._inner._run_for_source(source, targets, siblings)
        if not entries:
            return [], {}

        uncontested = [e for e in entries if e.alt_rel_type is None]
        contested = [e for e in entries if e.alt_rel_type is not None]

        if not contested:
            return entries, id_map

        node_by_id: dict[UUID, KBNode] = {n.id: n for n in [source, *targets]}
        try:
            tiebreaker_result = await self._tiebreak(contested, id_map, node_by_id)
        except _SelfReviewRetryExhaustedError:
            logger.warning(
                "%s: tiebreaker exhausted retries for source=%s — keeping originals",
                self.name,
                source,
            )
            return entries, id_map

        updated_contested = self._apply_tiebreaker(contested, tiebreaker_result, source)
        return uncontested + updated_contested, id_map

    async def _tiebreak(
        self,
        contested: list[E],
        id_map: dict[str, UUID],
        node_by_id: dict[UUID, KBNode],
    ) -> TiebreakerResult:
        relations_json = _contested_to_json(contested, id_map, node_by_id)
        messages = [
            SystemMessage(
                content=[{"type": "text", "text": self._system_prompt, "cache_control": {"type": "ephemeral"}}]
            ),
            HumanMessage(content=_TIEBREAKER_PROMPT.format(relations_json=relations_json)),
        ]
        return await self._inner._retryable_structured_ainvoke(
            messages=messages,
            response_model=TiebreakerResult,
            raise_on_exhaustion=_SelfReviewRetryExhaustedError(f"{self.name} tiebreaker exhausted retries"),
            on_error_log_prefix=f"{self.name}.tiebreaker",
            llm=self._review_llm,
        )

    def _apply_tiebreaker(
        self,
        contested: list[E],
        result: TiebreakerResult,
        source: KBNode,
    ) -> list[E]:
        if len(result.decisions) != len(contested):
            logger.warning(
                "%s: tiebreaker count mismatch (%d decisions / %d contested) for source=%s — keeping originals",
                self.name,
                len(result.decisions),
                len(contested),
                source,
            )
            return contested

        updated = []
        for entry, decision in zip(contested, result.decisions, strict=True):
            try:
                chosen = RelationshipType(decision.chosen)
                entry.rel_type = chosen
                logger.debug(
                    "%s: tiebreaker chose %s over %s for source=%s (rationale: %s)",
                    self.name,
                    decision.chosen,
                    entry.alt_rel_type,
                    source,
                    decision.rationale,
                )
            except ValueError:
                logger.warning(
                    "%s: tiebreaker returned invalid rel_type %r for source=%s — keeping original %s",
                    self.name,
                    decision.chosen,
                    source,
                    entry.rel_type,
                )
            updated.append(entry)

        return updated


def _contested_to_json(
    contested: list[E],
    id_map: dict[str, UUID],
    node_by_id: dict[UUID, KBNode],
) -> str:
    rows = []
    for entry in contested:
        src_uuid = id_map[entry.source_id]
        tgt_uuid = id_map[entry.target_id]
        src_node = node_by_id[src_uuid]
        tgt_node = node_by_id[tgt_uuid]
        rows.append(
            {
                "source": {"title": src_node.title, "description": src_node.description, "type": src_node.type},
                "target": {"title": tgt_node.title, "description": tgt_node.description, "type": tgt_node.type},
                "rel_type": entry.rel_type,
                "alt_rel_type": entry.alt_rel_type,
                "rationale": entry.rationale,
            }
        )
    return json.dumps(rows, indent=2)
