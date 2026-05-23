<p align="center">
  <a href="README.md">简体中文</a> |
  <a href="README_en.md">English</a>
</p>

# LLM AIO Gateway · 多合一 LLM API 网关

> 统一的 OpenAI / Anthropic / Responses 三协议 API 网关，内置视觉模型注入——让任意文本模型也能"看见"图片。

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE)

---

## ✨ 功能特性

| 特性 | 说明 |
|---|---|
| 🖼️ **视觉模型注入** | **核心卖点** — 自动将用户发送的图片交给视觉模型描述，让任意非多模态模型(Deepseek/GLM等)也能"看到"图片内容，省去升级多模态模型的成本 |
| 🔄 三协议代理 | 同时支持 OpenAI Chat Completions、Anthropic Messages、OpenAI Responses 三种 API 协议 |
| 🎯 透明路由 | 基于用户 / 调用 API Key / 请求模型名的路由规则，支持 `*` 通配符，无缝切换上游模型 |
| ⚡ 工具调用断路器 | 连续纯工具调用超过阈值自动剥离 `tools`，避免模型陷入无限循环 |
| 📊 Web 管理面板 | 内置 SPA 管理界面，提供商 / 用户管理 / 调用 API Key / 路由规则一站式管理 + 实时调用统计 |
| 🔑 多租户 | 用户 + 调用 API Key 认证体系，每个 Key 独立模型白名单和用量追踪 |
| 🧠 上下文感知 | 透明处理 reasoning/thinking 内容的多轮缓存与回传，无需客户端干预 |

---

## 🖼️ 核心特性：视觉模型注入

### 解决的问题

大多数高性能低价模型（如 DeepSeek-V3、MiniMax-M*）**不支持图片输入**，迫使你必须升级到价格翻倍的多模态模型才能处理图文混合对话。视觉模型注入让你继续使用廉价文本模型，同时获得图片理解能力。

### 工作流程

```
用户发送图片                    网关自动处理                      模型收到文本
───────┴────────    ───────────────┴───────────────    ───────────┴──────────
  "这是什么?"      ① 提取图片 → 视觉模型生成描述           "这是什么?
   + 📷 photo.jpg    ② 描述注入到对话上下文                [Image #1: 一只橘猫
                    ③ 去掉图片，发送纯文本                    坐在窗台上...]"
```

1. 用户正常发送图文消息到 `/chat/completions` 或 `/messages`
2. 网关检测到图片，调用配置的视觉模型（如 MiniCPM-V、Qwen-VL）生成文字描述
3. 图片内容替换为 `[Image #N: 描述文本]`，发送给目标文本模型
4. 模型回复转发回客户端 —— **整个过程对客户端完全透明**

### 配置步骤

**第一步** — 在 `config.json` 中配置一个视觉模型（预处理器），该步骤也可在WebUI的“视觉模型注入-新增预处理器”中配置：

```json
{
  "preprocessors": {
    "my-vision": {
      "api_base": "http://127.0.0.1:8080/v1",
      "model": "MiniCPM-V-4.6",
      "api_key": "sk-xxx",
      "timeout": 60,
      "max_images": 20,
      "max_tokens": 1024,
      "prompt": "Please describe this image in detail.",
      "enabled": true
    }
  }
}
```

**第二步** — 在管理面板为需要此功能的模型开启开关：

- 进入 **视觉模型注入** → 在目标模型行点击开关（开/关）
- 或用 API：`PUT /admin/models/preprocessor` `{"model_id": "provider/model-name", "enabled": true}`

**完成。** 此后所有发给该模型的图片请求都会自动注入描述。

> 视觉注入的决策基于用户**请求的模型名**而非路由后的目标模型，确保路由规则修改不影响注入行为。

### 推荐的本地视觉模型

以下模型可在本地部署（vLLM / llama.cpp / Ollama），作为视觉注入前端：

| 模型 | HuggingFace | 特点 |
|------|------------|------|
| **MiniCPM-V 4.6** | [openbmb/MiniCPM-V-4.6](https://huggingface.co/openbmb/MiniCPM-V-4.6) | **速度快**，仅 1B 参数（SigLIP2-400M + Qwen3.5-0.8B），CPU推理可达35+ tokens/s，适合轻量场景 |
| **Qwen3.6-35B-A3B** | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | **效果好**，35B 总参 / 3B 激活，256 专家 MoE + 27 层 ViT，原生 262K 上下文，描述质量最高 |
---

除此之外您依然可以使用在线多模态模型作为其他模型的预处理器来实现视觉注入。

## 🚀 快速开始

### 环境要求

- Python 3.10+
- pip

### Docker 部署（推荐）

```bash
git clone https://github.com/PriceNing/LLM-AIO-Gateway.git
cd LLM-AIO-Gateway
docker compose up -d
```

首次启动自动在 `./data` 目录生成 `config.json` 和 `data.db`。服务运行在 `http://localhost:8000`。

更新时拉取最新代码后重建镜像：

```bash
git pull
docker compose up -d --build
```

### 手动安装

```bash
git clone https://github.com/PriceNing/LLM-AIO-Gateway.git
cd LLM-AIO-Gateway
pip install -r requirements.txt
python main.py
```

首次启动会自动在项目目录生成 `config.json`（含默认配置）和 `data.db`（SQLite 数据库）。

服务启动后访问 `http://localhost:8000`：

- **管理面板**：浏览器直接打开，首次访问提示创建管理员账号
- **API 端点**：`/chat/completions`、`/messages`、`/responses`、`/models`

### 接入提供商

1. 登录管理面板 → **提供商** → 新增提供商
2. 填写名称、API Base URL、上游 API Key、类型（OpenAI 兼容 / Anthropic 兼容）
3. 点击 **刷新** 自动从远端 `/models` 发现可用模型
4. **用户管理** → 新增用户 → 生成调用 API Key
5. 开始调用：

```bash
curl http://localhost:8000/chat/completions \
  -H "Authorization: Bearer sk-aio-xxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{"model": "provider/model-name", "messages": [{"role": "user", "content": "Hello"}]}'
```

---

## 📖 API 端点

所有代理端点同时挂载在根路径和 `/v1/` 前缀下，两种方式等效：

| 端点 | 协议 | 说明 |
|---|---|---|
| `POST /chat/completions` | OpenAI Chat | 流式 / 非流式聊天补全 |
| `POST /completions` | OpenAI Text | 文本补全 |
| `POST /messages` | Anthropic | 流式 / 非流式消息（含工具调用） |
| `POST /responses` | OpenAI Responses | Codex CLI / OpenAI Responses 格式 |
| `GET /models` | OpenAI | 可用模型列表 |

> 根路径示例：`http://localhost:8000/chat/completions`
> `/v1/` 前缀示例：`http://localhost:8000/v1/chat/completions`

### 管理 API

| 端点 | 说明 |
|---|---|
| `POST /auth/login` | 管理员登录 |
| `GET/POST /admin/providers` | 提供商管理 |
| `POST /admin/providers/{id}/refresh` | 从远端 `/models` 自动发现模型 |
| `GET/POST /admin/users` | 用户管理 |
| `POST/PUT/DELETE /admin/users/{username}/api-keys` | 调用 API Key 管理 |
| `GET/POST /admin/routing-rules` | 路由规则管理 |
| `GET /admin/stats` | 调用统计 |
| `PUT /admin/models/preprocessor` | 切换模型视觉注入开关 |

---

## ⚙️ 配置参考

`config.json` 首次启动自动生成，修改后需重启服务生效。

### 顶层设置

| 键 | 默认值 | 说明 |
|---|---|---|
| `host` | `"0.0.0.0"` | 监听地址 |
| `port` | `8000` | 监听端口 |
| `database` | `"data.db"` | SQLite 数据库路径 |

### `defaults` 节

| 键 | 默认值 | 说明 |
|---|---|---|
| `max_tokens` | `16384` | 客户端未传 `max_tokens` 时的默认值 |
| `temperature` | `0.7` | 默认温度 |
| `tool_only_limit` | `20` | 连续工具调用断路器阈值 |
| `min_image_max_tokens` | `2000` | 含图片请求的 max_tokens 下限 |
| `session_ttl_hours` | `12` | 管理员会话有效期 |
| `request_log_max` | `200` | 内存请求日志保留条数 |
| `reasoning_cache_ttl` | `1800` | reasoning 缓存 TTL（秒） |
| `reasoning_cache_max_size` | `1000` | reasoning 缓存最大条目 |
| `tool_only_turns_ttl` | `600` | 工具调用计数器 TTL（秒） |
| `tool_only_turns_max_size` | `2000` | 工具调用计数器最大条目 |
| `image_cache_max_size` | `500` | 图像描述缓存最大条目 |

### `preprocessors` 节

配置用于图片描述的视觉模型（预处理器），每个条目包含 `api_base`、`model`、`api_key`、`timeout`、`max_images`、`prompt`、`max_tokens`、`enabled` 等字段。

### 环境变量

| 变量 | 说明 |
|---|---|
| `LLM_GATEWAY_CONFIG` | 覆盖 `config.json` 文件路径 |

---

## 🏗️ 架构

```
Client (Claude code / Codex / OpenCode / OpenWebUI / curl)
       │
       ▼
┌─────────────────────────────┐
│     LLM AIO Gateway :8000   │
│  ┌──────────┐ ┌───────────┐ │
│  │ proxy    │ │ admin/    │ │
│  │ router   │ │ auth      │ │
│  └────┬─────┘ └─────┬─────┘ │
│       │             │       │
│  ┌────┴─────────────┴────┐  │
│  │   liteLLM + patches   │  │
│  │   vision injector     │  │
│  │   routing engine      │  │
│  └───────────┬───────────┘  │
│              │              │
│  ┌───────────┴───────────┐  │
│  │    SQLite data.db     │  │
│  │    TTLDict caches     │  │
│  └───────────────────────┘  │
└─────────────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│  Upstream LLM Providers      │
└──────────────────────────────┘
```

- **存储**：SQLite（提供商、用户、调用 API Key、路由规则、统计）
- **缓存**：进程内 `TTLDict`（线程安全，TTL + 容量淘汰）
- **配置**：`config.json` 自动生成 + `LLM_GATEWAY_CONFIG` 环境变量覆盖路径
- **流式**：后台线程桥接同步 liteLLM → 异步 SSE，客户端断开自动取消上游请求

---

## ✅ 已测试平台

以下平台已在生产环境中验证通过。

### 上游提供商

| 提供商 | 协议类型 | 已测试模型 |
|--------|---------|-----------|
| **DeepSeek** | OpenAI 兼容 / Anthropic 兼容 | V4-Flash, V4-Pro |
| **MiniMax** | OpenAI 兼容 / Anthropic 兼容 | M2.7 |
| **OpenCode Go** | OpenAI 兼容 | hy3-preview, kimi-2.6 |

### 下游 Agent / 客户端

| Agent | 使用端点 | 说明 |
|--------|---------|------|
| **Claude Code CLI** | `/messages` | Anthropic 原生协议，支持工具调用多轮回传 |
| **OpenCode CLI** | `/chat/completions`, `/responses` | 开源编程 Agent，视觉注入 + 路由规则联动；多模态需在 `opencode.json` 中为目标模型声明 `modalities: {input: ["text", "image"]}` |
| **Codex Desktop** | `/responses` | OpenAI Codex 桌面客户端，透明接入任意模型 |
| **OpenWebUI** | `/chat/completions` | OpenWebUI网页聊天 |

### OpenCode 多模态配置

OpenCode CLI 默认认为自定义模型不支持图片输入。如需配合视觉模型注入使用，须在 `opencode.json` 中为目标模型声明多模态能力：

```json
"models": {
    "your-model": {
        "name": "your-model",
        "limit": {"output": 4096, "context": 128000},
        "modalities": {"input": ["text", "image"], "output": ["text"]}
    }
}
```

---

## 📝 许可证

MIT — 详见 [LICENSE](LICENSE) 文件。
