from langchain_core.language_models import BaseChatModel

from seshat.agents.identification.action_item import ActionItemIdentificationAgent
from seshat.agents.identification.base import _BaseIdentificationAgent
from seshat.agents.identification.decision import DecisionIdentificationAgent
from seshat.agents.identification.open_question import OpenQuestionIdentificationAgent
from seshat.agents.identification.reflective import ReflectiveIdentificationAgent
from seshat.agents.identification.risk import RiskIdentificationAgent
from seshat.config.settings import ExtractionConfig
from seshat.models.enums import ConceptType
from seshat.utils.hashing import fingerprint
from seshat.utils.log import get_logger

logger = get_logger(__name__)


class IdentificationAgentRegistry:
    def __init__(self, llm: BaseChatModel, config: ExtractionConfig, review_llm: BaseChatModel | None = None) -> None:
        self._agents: dict[ConceptType, _BaseIdentificationAgent] = {
            concept_type: _make_agent(agent_cls, llm, config, review_llm)
            for concept_type, agent_cls in (
                (ConceptType.ACTION_ITEM, ActionItemIdentificationAgent),
                (ConceptType.DECISION, DecisionIdentificationAgent),
                (ConceptType.OPEN_QUESTION, OpenQuestionIdentificationAgent),
                (ConceptType.RISK, RiskIdentificationAgent),
            )
        }

    def get(self, concept_type: ConceptType) -> _BaseIdentificationAgent:
        agent = self._agents.get(concept_type)
        if agent is None:
            raise KeyError(f"No agent registered for {concept_type}")
        return agent

    def fingerprint(self) -> str:
        """8-char hex digest of all agents' fingerprints concatenated.

        All four concept types always fire per example (parallel fan-out in the orchestrator),
        so any prompt change (including the validate prompt) busts the full identification cache.
        """
        combined = "".join(agent.fingerprint() for agent in self._agents.values())
        return fingerprint(combined)

    def prompt_texts(self) -> dict[str, str]:
        texts = {}
        for concept_type, agent in self._agents.items():
            for prompt_type, prompt in agent.prompt_texts().items():
                texts[f"{concept_type}_{prompt_type}"] = prompt
        return texts


def _make_agent(
    agent_cls: type[_BaseIdentificationAgent],
    llm: BaseChatModel,
    config: ExtractionConfig,
    review_llm: BaseChatModel | None,
) -> _BaseIdentificationAgent:
    inner = agent_cls(
        llm=llm, config=config.identification, grouped_identification_types=config.grouped_identification_types
    )
    if not config.identification_self_review.enabled:
        return inner

    logger.debug("Using Reflective%s", agent_cls.__name__)
    return ReflectiveIdentificationAgent(inner=inner, review_llm=review_llm or llm)
