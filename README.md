<p align="center">
  <a href="README.md">简体中文</a> |
  <a href="README_en.md">English</a>
</p>

# LLM AIO Gateway - 多合一 LLM API 网关

LLM AIO Gateway 是一个基于 FastAPI 的统一 LLM API 网关，用一个服务代理多个 OpenAI 兼容和 Anthropic 兼容的上游提供商。它同时暴露 OpenAI Chat Completions、旧版 Completions、Anthropic Messages、OpenAI Responses 等接口，并提供路由规则、API Key 管理、reasoning 连续性、工具调用修复和视觉模型注入。

当前代理核心基于统一内部消息格式：所有客户端协议先转换为 `InternalRequest` / `InternalMessage`，再进入共享策略层。策略层会产出结构化 `RoutingDecision`，随后由上游适配器发送给 OpenAI/liteLLM 或 Anthropic 兼容接口，最后再渲染回客户端需要的协议格式。

## 功能特性

| 功能 | 说明 |
|---|---|
| 统一协议网关 | 支持 `/chat/completions`、`/completions`、`/messages`、`/responses`、`/models`，同时挂载在根路径和 `/v1` 前缀。 |
| OpenAI / Anthropic 提供商 | OpenAI 兼容提供商走 liteLLM；Anthropic 兼容提供商从内部 IR 直接投影到 Anthropic Messages 请求。 |
| 共享 IR 流程 | 路由、预处理、reasoning 缓存、工具修复、断路器都运行在统一内部消息上，避免端点各自维护转换逻辑。 |
| 结构化路由 | 策略层返回 `RoutingDecision`，记录 requested/resolved/target model、目标 provider、命中的规则和原因。 |
| 视觉模型注入 | 图片可先由配置的视觉模型生成描述，再替换为文本，让纯文本模型也能处理图片上下文。 |
| 工具调用可靠性 | 保留工具调用 ID，修复 malformed JSON 工具参数，并提供工具调用循环断路器。 |
| Reasoning 连续性 | 对 DeepSeek 等 thinking 模型自动缓存和回传 `reasoning_content`，保证多轮工具调用不中断。 |
| Web 管理面板 | 管理提供商、用户、API Key、路由规则、模型预处理器和调用统计。 |
| SQLite 存储 | 提供商、用户、密钥、路由规则、统计和请求记录保存在 `data.db`。 |

## 快速开始

### 环境要求

- Python 3.10+
- pip

### 手动安装

```bash
pip install -r requirements.txt
python main.py
```

默认服务地址为 `http://localhost:8000`。首次启动会自动创建 `config.json` 和 `data.db`。

### Docker

```bash
docker compose up -d
```

## 首次配置

1. 打开 `http://localhost:8000`。
2. 创建第一个管理员账号。
3. 在管理面板添加上游提供商。
4. 刷新模型列表，或手动添加模型。
5. 创建普通用户并生成 API Key。
6. 使用 `Authorization: Bearer sk-aio-...` 调用网关。

## 提供商类型

| 类型 | 上游接口 | 网关行为 |
|---|---|---|
| `openai` | OpenAI 兼容接口 | 内部请求投影为 OpenAI Chat 格式，经 liteLLM 调用。 |
| `anthropic` | Anthropic Messages 兼容接口 | 内部请求投影为原生 Anthropic Messages 格式，直接请求 `{api_base}/v1/messages`。 |

模型 ID 支持 `provider/model` 和简单 `model` 两种形式。复合形式会直接定位到指定提供商；简单模型名会选择第一个启用且匹配的提供商模型。

## API 端点

所有代理端点同时支持根路径和 `/v1` 前缀。

| 端点 | 协议 | 说明 |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions | 对话、工具、流式、图片输入。适合 OpenCode / OpenWebUI 等客户端。 |
| `POST /v1/completions` | OpenAI 旧版 Completions | `prompt` 会包装成内部 user message，再渲染为 `choices[0].text`。 |
| `POST /v1/messages` | Anthropic Messages | Claude Code 兼容 Messages API，支持工具、流式、图片。 |
| `POST /v1/responses` 或 `/responses` | OpenAI Responses | Codex 兼容 Responses API，支持工具、流式、previous response ID。 |
| `GET /v1/models` | OpenAI Models | 返回当前 API Key 可用模型列表。 |

### Chat Completions 示例

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-aio-xxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "provider/model-name",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 1024
  }'
```

### Completions 示例

```bash
curl http://localhost:8000/v1/completions \
  -H "Authorization: Bearer sk-aio-xxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "provider/model-name",
    "prompt": "写一首关于春天的短诗。",
    "max_tokens": 200
  }'
```

### Anthropic Messages 示例

```bash
curl http://localhost:8000/v1/messages \
  -H "Authorization: Bearer sk-aio-xxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "provider/model-name",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": [{"type": "text", "text": "你好"}]}]
  }'
```

### Responses 示例

```bash
curl http://localhost:8000/v1/responses \
  -H "Authorization: Bearer sk-aio-xxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "provider/model-name",
    "input": "你好"
  }'
```

## 视觉模型注入

视觉模型注入用于让纯文本模型处理图片输入。为请求模型启用预处理器后，网关会将当前轮次图片交给配置的视觉模型描述，剥离原始图片数据，并把描述文本注入对话。

在 `config.json` 中配置预处理器：

```json
{
  "preprocessors": {
    "my-vision": {
      "api_base": "http://127.0.0.1:8080/v1",
      "model": "MiniCPM-V-4.6",
      "api_key": "",
      "timeout": 60,
      "max_images": 20,
      "max_tokens": 1024,
      "prompt": "Please describe this image in detail.",
      "enabled": true
    }
  }
}
```

然后在管理面板为目标模型开启预处理器。是否预处理取决于用户请求的原始模型，而不是路由后的目标模型。

## 路由规则

路由规则可按用户名、API Key 子串、请求模型通配符透明重定向请求。第一条匹配规则生效。

路由发生在共享策略层，结果以 `RoutingDecision` 表达。日志会记录原始请求模型、解析后的模型、目标模型、目标提供商、命中规则和原因，便于排查“为什么请求去了这个上游”。

规则结构：

```json
{
  "name": "route-example",
  "enabled": true,
  "username": "",
  "api_key_pattern": "",
  "match_model": "MiniMax-M2*",
  "target_model": "target-model",
  "target_provider": "target-provider"
}
```

## 配置

`config.json` 保存服务级配置，修改后需要重启服务。

重要默认项：

| 键 | 默认值 | 说明 |
|---|---:|---|
| `max_tokens` | 16384 | 客户端未传 `max_tokens` 和 `max_completion_tokens` 时使用。 |
| `temperature` | 0.7 | 默认温度。 |
| `tool_only_limit` | 20 | 工具调用循环断路器阈值。 |
| `min_image_max_tokens` | 2000 | 含图片请求的最小 max tokens。 |
| `reasoning_cache_ttl` | 1800 | reasoning 缓存 TTL，单位秒。 |
| `reasoning_cache_max_size` | 1000 | reasoning 缓存容量。 |
| `tool_only_turns_ttl` | 600 | 工具调用计数 TTL，单位秒。 |
| `tool_only_turns_max_size` | 2000 | 工具调用计数容量。 |
| `image_cache_max_size` | 500 | 图片描述缓存容量。 |

## 架构摘要

```text
客户端端点
  -> 协议入口
  -> 内部 IR
  -> 共享策略层（RoutingDecision、预处理、reasoning、工具修复）
  -> OpenAI/liteLLM 适配器或 direct Anthropic 适配器
  -> 内部输出
  -> 协议出口
```

端点特有的协议细节只保留在 ingress/egress。路由、预处理、reasoning 缓存、工具修复和适配器选择都基于统一内部格式运行。

主要代码边界：

| 模块 | 职责 |
|---|---|
| `app/router/proxy.py` | FastAPI 端点、鉴权、provider 解析、adapter 调度、非流式请求统计。 |
| `app/protocols/ingress.py` | `/chat/completions`、`/completions`、`/messages`、`/responses` 请求体转换为内部 IR。 |
| `app/core/policy.py` | 路由决策、消息规范化、预处理挂钩、reasoning 注入、工具参数修复、tool-only 限制。 |
| `app/core/state.py` | TTL cache、reasoning cache、tool-only counter、response chain cache。 |
| `app/core/streaming.py` | 流式事件计量、reasoning 存储、tool-only 计数、流式错误渲染和统计回调。 |
| `app/core/images.py` | data URI 图片提取、图片内容检测和 OpenAI 图像内容归一化。 |
| `app/adapters/` | 将内部请求投递给 OpenAI/liteLLM 或 direct Anthropic Messages，并转回内部输出事件。 |
| `app/protocols/egress.py` | 将内部输出渲染回 Chat、Completions、Messages、Responses 协议。 |
| `app/services/lite_llm.py` | 仅作为 OpenAI 兼容上游的 liteLLM wrapper 和最小 reasoning 兼容补丁。 |

## 测试

```bash
pytest tests/ -q
```

当前预期结果：`298 passed`。

真实烟测建议：

- Claude Code -> `/messages`
- Codex -> `/responses`
- OpenCode -> `/chat/completions`
- curl -> `/completions`
- 每类端点至少测一个 OpenAI 兼容提供商和一个 Anthropic 兼容提供商。

## 许可证

MIT。详见 `LICENSE` 文件。
