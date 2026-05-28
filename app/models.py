from pydantic import BaseModel, Field
from typing import Optional

class ModelInfo(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9._-]+$")
    name: str
    enabled: bool = True

class ProviderBase(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9._-]+$")
    name: str
    provider_type: str = Field(pattern="^(openai|anthropic)$")
    api_base: str
    api_key: str
    enabled: bool = True
    models: list[ModelInfo] = Field(default_factory=list)
    extra_headers: dict = Field(default_factory=dict)

class ProviderCreate(ProviderBase):
    pass

class ProviderUpdate(BaseModel):
    name: Optional[str] = None
    provider_type: Optional[str] = Field(default=None, pattern="^(openai|anthropic)$")
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    enabled: Optional[bool] = None
    models: Optional[list[ModelInfo]] = None
    extra_headers: Optional[dict] = None

class StatsResponse(BaseModel):
    total_calls: int
    failed_calls: int
    success_rate: float
    last_reset: str
    stats_by_model: dict = Field(default_factory=dict)
    request_log: list = Field(default_factory=list)
    users: list = Field(default_factory=list)
    timeline: dict = Field(default_factory=dict)
    distribution: dict = Field(default_factory=dict)
    timeline_models: dict = Field(default_factory=dict)
