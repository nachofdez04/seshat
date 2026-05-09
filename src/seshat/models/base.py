from pydantic import BaseModel, ConfigDict


class SeshatModel(BaseModel):
    """Frozen base for all immutable domain-fact models."""

    model_config = ConfigDict(frozen=True)
