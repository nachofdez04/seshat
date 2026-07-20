from enum import StrEnum, auto


class ConceptType(StrEnum):
    DECISION = auto()
    RISK = auto()
    ACTION_ITEM = auto()
    OPEN_QUESTION = auto()


class RelationshipType(StrEnum):
    MITIGATES = auto()
    BLOCKS = auto()
    CONFLICTS_WITH = auto()
    DEPENDS_ON = auto()
    SUPERSEDES = auto()
    AMENDS = auto()
    RESOLVES = auto()


class NodeStatus(StrEnum):
    APPROVED = auto()
    PENDING_REVIEW = auto()
    REJECTED = auto()


class NodeState(StrEnum):
    CURRENT = auto()
    AMENDED = auto()
    SUPERSEDED = auto()


class ApprovalMethod(StrEnum):
    INDIVIDUAL = auto()
    BULK = auto()
    AUTO = auto()
    THRESHOLD = auto()
    MANUAL = auto()


class IngestionSource(StrEnum):
    PIPELINE = auto()
    INIT = auto()
    MANUAL = auto()


class RelationshipSource(StrEnum):
    PIPELINE = auto()
    MANUAL = auto()
    INIT = auto()


class JobStatus(StrEnum):
    PENDING = auto()
    TRANSCRIBING = auto()
    IDENTIFYING = auto()
    AWAITING_REVIEW = auto()
    RESOLVING = auto()
    WRITING = auto()
    DONE = auto()
    FAILED = auto()

    @classmethod
    def terminal_statuses(cls) -> tuple:
        return (cls.DONE, cls.FAILED)

    @property
    def is_terminal(self) -> bool:
        return self in self.__class__.terminal_statuses()

    @classmethod
    def running_statuses(cls) -> tuple:
        return (cls.TRANSCRIBING, cls.IDENTIFYING, cls.RESOLVING, cls.WRITING)

    @property
    def is_running(self) -> bool:
        return self in self.__class__.running_statuses()

    @classmethod
    def stranded_statuses(cls) -> tuple:
        return (cls.RESOLVING, cls.WRITING)

    @property
    def is_stranded(self) -> bool:
        return self in self.__class__.stranded_statuses()


class LLMProvider(StrEnum):
    OPENAI = auto()
    ANTHROPIC = auto()
    AZURE_OPENAI = auto()
    BEDROCK_CONVERSE = auto()


class TranscriptionProvider(StrEnum):
    ASSEMBLYAI = auto()
    OPENAI = auto()
    DEEPGRAM = auto()


class VectorStoreProvider(StrEnum):
    PGVECTOR = auto()


class EmbeddingProvider(StrEnum):
    OPENAI = auto()
    AZURE_OPENAI = auto()
    ANTHROPIC = auto()


class SecretsProvider(StrEnum):
    ENV = auto()
    AWS = auto()


class DocumentLoaderProvider(StrEnum):
    MARKDOWN = auto()


class CallType(StrEnum):
    LLM_INPUT = auto()
    LLM_OUTPUT = auto()
    EMBEDDING = auto()
    TRANSCRIPTION = auto()


class GraphDirection(StrEnum):
    INBOUND = auto()
    OUTBOUND = auto()
    BOTH = auto()


class SearchMode(StrEnum):
    SEMANTIC = auto()
    KEYWORD = auto()
    HYBRID = auto()


class UserRole(StrEnum):
    # Ordered lowest to highest — definition order determines rank.
    VIEWER = auto()
    REVIEWER = auto()
    OPERATOR = auto()
    ADMIN = auto()

    def is_at_least(self, minimum: "UserRole") -> bool:
        """Return True if this role meets or exceeds minimum."""
        members = list(UserRole)
        return members.index(self) >= members.index(minimum)


class HealthStatus(StrEnum):
    OK = auto()
    DEGRADED = auto()
    ERROR = auto()
