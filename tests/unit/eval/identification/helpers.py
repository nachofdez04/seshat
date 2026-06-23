from seshat.eval.models import IdentificationCorpusNode
from seshat.models.enums import ConceptType


def corpus_node(
    quote: str,
    ctype: ConceptType,
    title: str = "T",
    description: str = "A description.",
    extra_fields: dict | None = None,
) -> IdentificationCorpusNode:
    return IdentificationCorpusNode(
        quote=quote,
        type=ctype,
        title=title,
        description=description,
        extra_fields=extra_fields or {},
    )
