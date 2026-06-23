from typing import ClassVar

import spacy
import spacy.tokens


class HeuristicsScorer:
    # fmt: off
    _HEDGING_TOKENS: ClassVar[set[str]] = {"should", "might", "could", "may", "would", "possibly", "probably", "potentially", "maybe", "perhaps"}  # noqa: E501
    _FUTURE_AUX: ClassVar[set[str]] = {"will", "shall"}

    _TECH_PATTERNS: ClassVar[set[str]] = {
        # databases
        "postgresql", "postgres", "mysql", "mongodb", "redis", "elasticsearch", "sqlite", "dynamodb", "cassandra", "neo4j",  # noqa: E501
        # messaging / streaming
        "kafka", "rabbitmq", "celery", "airflow",
        # infra / orchestration
        "kubernetes", "docker", "terraform", "nginx", "ansible",
        # cloud
        "aws", "gcp", "azure",
        # frameworks / libs
        "fastapi", "django", "flask", "react", "vue", "typescript", "sqlalchemy", "pydantic", "langchain", "pytest",
        # observability
        "prometheus", "grafana", "mlflow",
        # programming languages
        "python", "golang", "rust", "java", "kotlin", "scala", "javascript", "ruby",
    }
    # fmt: on

    # Weights and saturation constants below are hand-tuned starting points with no empirical basis.
    # Before enabling grounding (which carries a 0.70 weight), these should be calibrated against
    # a labeled corpus using held-out precision/recall metrics per concept type.
    _W_QUOTE: float = 0.3
    _W_TITLE: float = 0.3
    _W_DESC: float = 0.4
    _QUOTE_WORD_SATURATION: int = 35

    # weights for title specificity subcomponents
    _W_TITLE_ENTITY: float = 0.45
    _W_TITLE_QUALIFIER: float = 0.35
    _W_TITLE_WORDS: float = 0.20
    _TITLE_WORD_SATURATION: int = 8

    def __init__(self, nlp: spacy.language.Language | None = None) -> None:
        self._nlp = nlp

    @property
    def nlp(self) -> spacy.language.Language:
        if self._nlp is None:
            self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    def score(self, source_quote: str, title: str, description: str) -> float:
        quote_score = min(len(source_quote.split()) / self._QUOTE_WORD_SATURATION, 1.0)
        title_score = self._title_specificity(title)
        desc_score = self._directness(description)
        return self._W_QUOTE * quote_score + self._W_TITLE * title_score + self._W_DESC * desc_score

    def _title_specificity(self, title: str) -> float:
        if not title.strip():
            return 0.0

        doc = self.nlp(title)
        word_score = min(len(title.split()) / self._TITLE_WORD_SATURATION, 1.0)
        entity_score = 1.0 if self._has_named_entity(doc) else 0.0
        qualifier_score = 1.0 if self._has_qualifier(doc) else 0.0
        return (
            self._W_TITLE_ENTITY * entity_score
            + self._W_TITLE_QUALIFIER * qualifier_score
            + self._W_TITLE_WORDS * word_score
        )

    def _directness(self, description: str) -> float:
        if not description.strip():
            return 0.0

        doc = self.nlp(description)
        score = 1.0

        root = next((t for t in doc if t.dep_ == "ROOT"), None)
        if root is None:
            return score

        children = set(root.children)
        child_lemmas = {t.lemma_.lower() for t in children}

        penalties: list[tuple[float, bool]] = [
            # hedging scoped to main clause children only, not subordinates
            (0.5, bool(child_lemmas & self._HEDGING_TOKENS)),
            # passive voice — both OntoNotes (en_core_web_sm) and Universal Dependencies label styles
            (0.75, any(t.dep_ in ("nsubjpass", "nsubj:pass", "auxpass", "aux:pass") for t in doc)),
            # future tense: planned, not yet decided
            (0.75, any(t.lemma_.lower() in self._FUTURE_AUX and t.dep_ == "aux" for t in children)),
            # no object/complement: vague assertion ("we agreed" vs "we agreed on Redis")
            (0.75, not any(t.dep_ in ("dobj", "obj", "attr") or (t.dep_ == "prep" and t.head == root) for t in doc)),
        ]

        for multiplier, applies in penalties:
            if applies:
                score *= multiplier
        return score

    def _has_named_entity(self, doc: spacy.tokens.Doc) -> bool:
        # domain-specific patterns are more reliable than en_core_web_sm NER for tech terms
        if any(t.text.lower() in self._TECH_PATTERNS for t in doc):
            return True

        # proper nouns (PROPN): spaCy tags specific names like "Redis", "Kubernetes", "Alice"
        if any(t.pos_ == "PROPN" for t in doc):
            return True

        # NER as fallback for org/product names not in _TECH_PATTERNS
        return any(ent.label_ in ("ORG", "PRODUCT") for ent in doc.ents)

    def _has_qualifier(self, doc: spacy.tokens.Doc) -> bool:
        root = next((t for t in doc if t.dep_ == "ROOT"), None)
        if root is None:
            return False

        for token in doc:
            # prep on root verb: "use Redis for session storage"
            # prep on root noun: "Database choice for session storage", "Policy on data retention"
            if token.dep_ == "prep" and token.head == root and root.pos_ in ("VERB", "NOUN"):
                return True
            # adverbial clause on root: "use Redis when traffic spikes", "deploy instead of migrating"
            if token.dep_ == "advcl" and token.head == root:
                return True

        return False
