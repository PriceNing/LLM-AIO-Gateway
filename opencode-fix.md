# OpenCode 修复记录

**日期**: 2026-05-16
**来源**: Sisyphus 代码审计 + Codex 交叉验证

---

## 修复清单（共 12 项）

### 🔴 P0 — 严重 Bug 修复

| # | 文件 | 行 | 问题 | 修复 |
|---|------|-----|------|------|
| 1 | `app/database.py` | 478 | `find_provider_by_model()` 无 `ORDER BY`，同名模型跨 provider 时路由结果不确定 | 添加 `ORDER BY p.id` |
| 2 | `app/router/proxy.py` | 238-251 | `_conversation_cache_key()` 纯图片对话指纹为空字符串，所有纯图片对话共享同一缓存键（MD5 碰撞） | 文本提取为空时降级为 `json.dumps(content_list)` 作为指纹 |
| 3 | `app/router/proxy.py` | 938-941 | `_stream_responses()` 缺少 `TOOL_ONLY_LIMIT` 断路保护（其他两个端点均有） | 在 streaming 开始前添加与其他端点一致的断路检查 |

### 🟡 P1 — 中等修复

| # | 文件 | 行 | 问题 | 修复 |
|---|------|-----|------|------|
| 4 | `app/security.py` | 60-80 | `_sessions` dict 无过期清理，从未被访问的 session 永久驻留内存 | 添加 daemon 线程每 5 分钟清理过期 session |
| 5 | `main.py` | 56 | CORS `allow_credentials=True` 与 `allow_origins=["*"]` 冲突（浏览器规范禁止此组合） | `allow_credentials` 改为 `False` |
| 6 | `CLAUDE.md` | 52, 100 | 第 52 行已更新为准确描述，但第 100 行仍保留过时的 "No app/security.py" 声明，两处矛盾 | 更新第 52 行为准确描述，删除第 100 行 |

### 🟢 P2 — 代码质量修复

| # | 文件 | 行 | 问题 | 修复 |
|---|------|-----|------|------|
| 7 | `app/database.py` | 439 | `update_provider()` 用内联元组拼接 f-string SQL 列名，安全扫描器告警 | 改用显式 `_updatable` set 变量 |
| 8 | `app/router/auth.py` | 21 | `require_admin_session` 对纯内存操作（dict 查找）使用 `asyncio.to_thread`，无效开销 | 移除 `asyncio.to_thread`，直接同步调用 |
| 9 | `app/services/lite_llm.py` | 39, 71 | except 块内重复 `import logging`（模块顶部已导入） | 删除 except 块内的重复 import |
| 10 | `app/services/lite_llm.py` | 209 | `build_completion_args()` 函数内 `from app.services.logger import get_logger` | 移至文件顶部 import 区域 |
| 11 | `app/database.py` | 107, 38-42 | `provider_models` 表缺少 `created_at` 字段（其他表均有） | 添加列 + `ALTER TABLE` 自动迁移兼容旧库 |

---

## 测试结果

```
18 passed in 8.68s
```

所有修改均通过现有测试套件，无回归。

---

## 改动文件

| 文件 | 改动行数 |
|------|---------|
| `app/database.py` | +8 |
| `app/router/proxy.py` | +12 |
| `app/security.py` | +21 |
| `main.py` | ±1 |
| `CLAUDE.md` | +1 / -4 |
| `app/router/auth.py` | ±1 |
| `app/services/lite_llm.py` | +1 / -4 |
