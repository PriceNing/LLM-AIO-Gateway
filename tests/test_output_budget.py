"""输出预算与入站体积默认值的行为测试。

对应两份现场反馈：
* 多轮带图对话撞 413（默认入站体积上限从 32 MiB 提到 128 MiB）。
* 客户端未指定 max_tokens 时，推理模型把思考内容与答案计入同一个 completion
  预算，16384 容易被思考吃光导致正文为空（按能力注入更大的推理预算）。
"""

import pytest

from app.config import default_config
from app.core.body_limit import DEFAULT_MAX_REQUEST_BODY_BYTES, max_request_body_bytes
from app.core.outcome import routing_details_from_policy
from app.core.policy import (
    RouteTarget,
    RoutingDecision,
    apply_output_budget_policy,
    prepare_request_policy,
)
from app.core.types import InternalRequest
from app.protocols.ingress import (
    anthropic_messages_to_internal,
    chat_completions_to_internal,
    completions_to_internal,
    responses_to_internal,
)


@pytest.fixture
def budget_defaults(monkeypatch):
    """把策略层读到的配置固定为代码默认值，测试不依赖仓库里的 config.json。"""
    from app.core import policy

    values = {"max_tokens": 16384, "reasoning_max_tokens": 32768}
    monkeypatch.setattr(policy, "get_default", lambda key, fallback=None: values.get(key, fallback))
    return values


def _request(**overrides) -> InternalRequest:
    fields = {
        "endpoint": "chat_completions",
        "requested_model": "m",
        "target_model": "m",
        "max_tokens": 16384,
    }
    fields.update(overrides)
    return InternalRequest(**fields)


# -- 入站体积上限 --

def test_default_body_limit_is_128_mib(monkeypatch):
    import app.core.body_limit as module

    monkeypatch.setattr(module, "get_default", lambda key, fallback=None: fallback)
    assert DEFAULT_MAX_REQUEST_BODY_BYTES == 128 * 1024 * 1024
    assert max_request_body_bytes() == 128 * 1024 * 1024


def test_config_default_matches_body_limit_constant():
    assert default_config()["defaults"]["max_request_body_bytes"] == DEFAULT_MAX_REQUEST_BODY_BYTES
    assert default_config()["defaults"]["reasoning_max_tokens"] == 32768


# -- 入口协议：记录客户端是否显式给出输出上限 --

def test_ingress_marks_unspecified_output_limit():
    req = chat_completions_to_internal({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert req.metadata["max_tokens_specified"] is False

    req = completions_to_internal({"model": "m", "prompt": "hi"})
    assert req.metadata["max_tokens_specified"] is False

    req = responses_to_internal({"model": "m", "input": "hi"})
    assert req.metadata["max_tokens_specified"] is False


def test_ingress_marks_specified_output_limit():
    req = chat_completions_to_internal({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 512,
    })
    assert req.metadata["max_tokens_specified"] is True
    assert req.max_tokens == 512

    req = chat_completions_to_internal({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "max_completion_tokens": 900,
    })
    assert req.metadata["max_tokens_specified"] is True
    assert req.max_tokens == 900

    req = anthropic_messages_to_internal({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 64,
    })
    assert req.metadata["max_tokens_specified"] is True
    assert req.max_tokens == 64

    # 显式 0 也是客户端意图（部分上游把 0/负值视为不限制），不得被覆盖。
    req = completions_to_internal({"model": "m", "prompt": "hi", "max_tokens": 0})
    assert req.metadata["max_tokens_specified"] is True
    assert req.max_tokens == 0


# -- 策略层：按能力注入输出预算 --

def test_reasoning_capability_raises_default_budget(budget_defaults):
    req = _request()
    assert apply_output_budget_policy(req, {"supports_reasoning": True}) == "reasoning"
    assert req.max_tokens == 32768


def test_non_reasoning_model_keeps_base_budget(budget_defaults):
    req = _request()
    assert apply_output_budget_policy(req, {"supports_tools": True}) == ""
    assert req.max_tokens == 16384

    req = _request()
    assert apply_output_budget_policy(req, None) == ""
    assert req.max_tokens == 16384


def test_client_specified_limit_is_never_touched(budget_defaults):
    req = _request(max_tokens=512, metadata={"max_tokens_specified": True})
    assert apply_output_budget_policy(req, {"supports_reasoning": True}) == ""
    assert req.max_tokens == 512


def test_client_requested_reasoning_counts_without_capability(budget_defaults):
    req = _request(extra={"reasoning_effort": "high"})
    assert apply_output_budget_policy(req, {}) == "reasoning"
    assert req.max_tokens == 32768

    req = _request(extra={"enable_thinking": True})
    assert apply_output_budget_policy(req, {}) == "reasoning"

    # 关闭思考的显式声明不算推理请求。
    req = _request(extra={"reasoning_effort": "none", "enable_thinking": False})
    assert apply_output_budget_policy(req, {}) == ""
    assert req.max_tokens == 16384


def test_budget_never_lowers_an_existing_value(budget_defaults):
    req = _request(max_tokens=65536)
    assert apply_output_budget_policy(req, {"supports_reasoning": True}) == ""
    assert req.max_tokens == 65536


def test_disabled_injection_is_respected(budget_defaults):
    # 管理员把 defaults.max_tokens 设为 0 = 网关不注入输出上限，推理预算同样不注入。
    budget_defaults["max_tokens"] = 0
    req = _request(max_tokens=0)
    assert apply_output_budget_policy(req, {"supports_reasoning": True}) == ""
    assert req.max_tokens == 0


def test_reasoning_default_disabled_falls_back_to_base(budget_defaults):
    budget_defaults["reasoning_max_tokens"] = 0
    req = _request()
    assert apply_output_budget_policy(req, {"supports_reasoning": True}) == ""
    assert req.max_tokens == 16384


# -- 策略编排：能力查询发生在路由之后 --

async def _noop_preprocess(request, model, provider_id, requested_model):
    return False


def _fake_conv_key(api_key, messages, previous_response_id):
    return f"{api_key}:{previous_response_id}:{len(messages)}"


@pytest.mark.asyncio
async def test_prepare_request_policy_uses_routed_target(monkeypatch, budget_defaults):
    from app.core import policy

    monkeypatch.setattr(policy, "apply_routing_rules", lambda *_: RoutingDecision(
        requested_model="source-model",
        resolved_model="source-model",
        target=RouteTarget(model="target-model", provider_id="target-provider"),
        matched=True,
        rule_id=3,
        rule_name="route-test",
        source="routing_rule",
        reason="test route",
    ))

    seen: list[tuple[str, str]] = []

    def capabilities_lookup(model: str, provider_id: str) -> dict:
        seen.append((model, provider_id))
        return {"supports_reasoning": True}

    req = chat_completions_to_internal({
        "model": "source-model",
        "messages": [{"role": "user", "content": "hi"}],
    })
    result = await prepare_request_policy(
        req,
        username="alice",
        api_key_value="key",
        preprocess_request=_noop_preprocess,
        conversation_cache_key=_fake_conv_key,
        normalize=False,
        preprocess=False,
        apply_ir_transforms=False,
        log_label="chat",
        capabilities_lookup=capabilities_lookup,
    )

    assert seen == [("target-model", "target-provider")]
    assert req.max_tokens == 32768
    assert result.output_budget == "reasoning"
    details = routing_details_from_policy(result)
    assert details["output_budget"] == "reasoning"


@pytest.mark.asyncio
async def test_prepare_request_policy_without_lookup_keeps_legacy_behavior(monkeypatch, budget_defaults):
    from app.core import policy

    monkeypatch.setattr(policy, "apply_routing_rules", lambda *_: RoutingDecision(
        requested_model="source-model",
        resolved_model="source-model",
        target=RouteTarget(model="source-model", provider_id=""),
    ))
    req = chat_completions_to_internal({
        "model": "source-model",
        "messages": [{"role": "user", "content": "hi"}],
    })
    result = await prepare_request_policy(
        req,
        username="alice",
        api_key_value="key",
        preprocess_request=_noop_preprocess,
        conversation_cache_key=_fake_conv_key,
        normalize=False,
        preprocess=False,
        apply_ir_transforms=False,
        log_label="chat",
    )
    assert req.max_tokens == 16384
    assert result.output_budget == ""
    assert "output_budget" not in routing_details_from_policy(result)
