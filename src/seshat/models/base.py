from typing import Any, Self

from pydantic import BaseModel, ConfigDict


class SeshatModel(BaseModel):
    """Frozen base for all immutable domain-fact models."""

    model_config = ConfigDict(frozen=True)

    def _with(self, **kwargs: Any) -> Self:
        return self.model_copy(update=kwargs)
