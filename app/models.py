from pydantic import BaseModel, Field
from typing import Optional

class ModelInfo(BaseModel):
    # Hidden/preview upstream model ids can contain '/', ':', '@' and spaces,
    # so only forbid control characters instead of allowlisting a narrow set.
    id: str = Field(min_length=1, max_length=200, pattern=r"^[^\x00-\x1f]+$")
    name: str
    enabled: bool = True
    source: Optional[str] = Field(default=None, pattern="^(auto|manual)$")

class ProviderBase(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9._-]+$")
    name: str
    provider_type: str = Field(pattern="^(openai|anthropic)$")
    api_base: str
    api_key: str
    enabled: bool = True
    models: list[ModelInfo] = Field(default_factory=list)
    extra_headers: dict = Field(default_factory=dict)
    request_timeout: int = Field(default=120, ge=1, le=3600)
    retry_count: int = Field(default=0, ge=0, le=10)
    retry_backoff: float = Field(default=0.5, ge=0, le=60)
    force_chat_completions: bool = False

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
    request_timeout: Optional[int] = Field(default=None, ge=1, le=3600)
    retry_count: Optional[int] = Field(default=None, ge=0, le=10)
    retry_backoff: Optional[float] = Field(default=None, ge=0, le=60)
    force_chat_completions: Optional[bool] = None

class StatsResponse(BaseModel):
    total_calls: int
    failed_calls: int
    degraded_calls: int = 0
    rejected_calls: int = 0
    cancelled_calls: int = 0
    stateful_fallback_blocked_calls: int = 0
    image_generation_calls: int = 0
    image_generation_failed_calls: int = 0
    image_generation_images: int = 0
    image_generation_bytes: int = 0
    success_rate: float
    health_rate: float = 100.0
    last_reset: str
    stats_by_model: dict = Field(default_factory=dict)
    request_log: list = Field(default_factory=list)
    users: list = Field(default_factory=list)
    timeline: dict = Field(default_factory=dict)
    distribution: dict = Field(default_factory=dict)
    timeline_models: dict = Field(default_factory=dict)
