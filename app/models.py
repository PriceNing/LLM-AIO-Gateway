from pydantic import BaseModel, Field
from typing import Optional, Any

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

class ProviderCreate(ProviderBase):
    pass

class ProviderUpdate(BaseModel):
    name: Optional[str] = None
    provider_type: Optional[str] = Field(default=None, pattern="^(openai|anthropic)$")
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    enabled: Optional[bool] = None
    models: Optional[list[ModelInfo]] = None

class ProviderResponse(ProviderBase):
    pass

class StatsResponse(BaseModel):
    total_calls: int
    failed_calls: int
    success_rate: float
    last_reset: str
    stats_by_model: dict = {}
    request_log: list = []
    users: list = []
    timeline: dict = {}
    distribution: dict = {}
    timeline_models: dict = {}

class ChatMessage(BaseModel):
    role: str
    content: Any

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list
    usage: dict
