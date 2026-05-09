from enum import StrEnum, auto


class ConceptType(StrEnum):
    ADR = auto()
    RISK = auto()
    AGREEMENT = auto()
    ACTION_ITEM = auto()


class RelationshipType(StrEnum):
    MITIGATES = auto()
    SUPPORTS = auto()
    CONFLICTS_WITH = auto()
    DEPENDS_ON = auto()
    SUPERSEDES = auto()
    AMENDS = auto()
    ASSIGNED_TO = auto()


class NodeStatus(StrEnum):
    AUTO_APPROVED = auto()
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


class IngestionSource(StrEnum):
    JOB = auto()
    INIT = auto()


class JobStatus(StrEnum):
    PENDING = auto()
    TRANSCRIBING = auto()
    EXTRACTING = auto()
    AWAITING_REVIEW = auto()
    WRITING = auto()
    DONE = auto()
    FAILED = auto()


class LLMProvider(StrEnum):
    OPENAI = auto()
    ANTHROPIC = auto()


class TranscriptionProvider(StrEnum):
    ASSEMBLYAI = auto()
    OPENAI = auto()
    DEEPGRAM = auto()


class VectorStoreProvider(StrEnum):
    PGVECTOR = auto()


class EmbeddingProvider(StrEnum):
    OPENAI = auto()
    ANTHROPIC = auto()
    COHERE = auto()
    FASTEMBED = auto()


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
