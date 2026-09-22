# 兼容补丁审计（存量债务盘点）

判定准则见 `AGENTS.md` → *Code Conventions → Compatibility Patch Boundary*（四问 + 明确禁止项）。
本文只记录**审计结论与证据**，不重复准则本身。

- 审计对象：`app/` 下所有"替客户端 / 替上游 / 替第三方库改写请求或响应"的兼容层
- 基线版本：`0.13.0`，`litellm==1.83.14`（requirements 精确锁定；生产容器实测同为 1.83.14）
- 测试基线：`1014 passed`

---

## 一、结论总表

| # | 位置 | 表面理由 | 真实归属 | 判定 | 状态 |
|---|---|---|---|---|---|
| 1 | `services/lite_llm.py` `_model_name_suggests_vision` + `model_cost` 注入 | 防 liteLLM 误广告视觉能力 | 对 `openai/` 前缀**完全无效**的死代码，且有污染 `model_cost` 的副作用 | **删除** | ✅ 本次已删 |
| 2 | `services/lite_llm.py` `_normalize_gpt5_temperature` / `_gpt5_temperature_supported` | 防 liteLLM 在本地拒绝 GPT-5 的 temperature | 真实模型约束（GPT-5 系只接受 `temperature=1`），但**用模型名硬编码在逻辑里** | **迁入能力层** | ✅ 已完成（§三） |
| 3 | `services/lite_llm.py` `_disable_thinking_when_tools_forced` | 注释写"DeepSeek 历史上拒绝 forced tool_choice + thinking" | 代码判断是通用形状（tools + forced tool_choice + thinking enabled），无厂商名 | **保留**（注释已满足 §六：写明替谁兜底 + 退出条件） | ✅ |
| 4 | `services/lite_llm.py` `_disable_thinking_for_missing_reasoning` | 历史 tool_call 缺 `reasoning_content` | 由 `router/proxy.py` 在**上游 400 之后**显式置位，属"事实驱动降级" | **保留** | ✅ |
| 5 | `services/lite_llm.py` `_system_messages_first` / `_merge_system_contents` | llama.cpp / Qwen 模板要求 system 首位 | OpenAI 协议本身要求单一前置 system，属协议规范化 | **保留** | ✅ |
| 6 | `services/lite_llm.py` 顶部 liteLLM `reasoning_content` 字段/转换器补丁 | 上游库丢字段 | 第三方库缺陷兜底，且已有 `compatibility_patch_status()` + 测试守护 | **保留**（纪律良好，可作范本） | ✅ |
| 7 | `core/images.py` `normalize_image_content` | 客户端把图片塞在文本里 | 网关主动改写，此前不校验完整性 → 自己造非法载荷 | **已收窄 + 加校验** | ✅ 本次已修 |
| 8 | `adapters/imagegen.py` `_is_grok_image_backend()`：`provider in {"grok","supergrok","xai"}` | 生图结果形态差异 | 厂商名硬编码 | **保留但登记**（见 §四） | ⏳ |
| 9 | `core/model_capabilities.py` builtin 家族表 | 离线兜底能力 | 能力层**本职**，名字集中且可被上层覆盖 | **保留**（设计如此） | ✅ |

---

## 二、本次已执行的改动

### 2.1 删除 `model_cost` 视觉注入（#1）

删除内容：`_model_name_suggests_vision()`（19 个模型名 marker）与 `build_completion_args()` 里的

```python
if (litellm_model.startswith("openai/")
        and litellm_model not in litellm.model_cost
        and _model_name_suggests_vision(litellm_model)):
    litellm.model_cost[litellm_model] = {"supports_vision": True}
```

证据链（五条，互相独立）：

1. **源码路径**：liteLLM 1.83.14 中 `litellm.supports_vision()` 的唯一非测试调用点是
   `llms/prompt_templates/factory.py` 的 `prompt_factory()` 中（`gemini` 分支）
   （`anthropic` 分支用另一条路径）。`openai/` provider 直接透传 `messages`，不走 prompt 转换。
2. **真实调用实验**：向不存在的上游发带 `image_url` 的 `openai/mimo-v2.6-pro` 请求，
   注册与不注册**结果一致**（均为 `APIError ... Connection error`，即请求已到达网络层，
   未被本地拦截）。
3. **测试依赖**：禁用该块后 `1014 passed`，无任何测试引用被删函数。
4. **网关自身不读**：全仓 `model_cost` / `get_model_info` / `get_max_tokens` 的引用**只有该块自身**。
5. **保留反而有害**：注入的残缺条目会让 `get_model_info` 返回
   `max_output_tokens: None`、`input_cost_per_token: 0`，任何未来依赖 `model_cost` 的代码
   都会读到被我们污染过的假信息。

### 2.2 图片载荷完整性校验（#7）

见 `CHANGELOG.md` [Unreleased] 与 `tests/test_image_payload_validation.py`。
要点：只提升能证明完整的位图、只改写 `user` 消息、只删除已提取的 URI。
属于"减少网关改写面"而非"让改写更聪明"。

---

## 三、已执行：GPT-5 temperature 迁入能力层（#2）

原代码在逻辑里硬编码模型名：

```python
def _gpt5_temperature_supported(model, kwargs):
    local_model = _local_litellm_model_name(model).lower()
    if not local_model.startswith("gpt-5"):        # ← 身份判断
        return True
    return local_model.startswith("gpt-5.1") and kwargs.get("reasoning_effort") in (None, "none")
```

每出一代模型就要改一次这段 `startswith`（现状已经打了第二代补丁），正是四问准则要拦的方向。

### 落地结果

| 层 | 改动 |
|---|---|
| `core/model_capabilities.py` | 新增 `_FLOAT_KEYS = ("fixed_temperature", "fixed_temperature_with_reasoning")` + `_FLOAT_LIMITS`（定义域 0.0–2.0），`normalize_capabilities` 增加浮点解析（拒绝 bool / 越界 / 无法解析，保持"未知=不锁"） |
| 同上（家族表） | `(("gpt-5.1",), {..., "fixed_temperature_with_reasoning": 1})` 排在 `(("gpt-5",), {..., "fixed_temperature": 1})` **之前** |
| `database.py` | 新增 `get_model_stored_capabilities(provider_id, model)`，读取 provider_models 上"上游透传 + 管理员覆盖"的合并结果 |
| `services/lite_llm.py` | 删除 `_local_litellm_model_name` / `_gpt5_temperature_supported` / `_normalize_gpt5_temperature`；改为 `model_temperature_locks()`（解析能力）+ `apply_temperature_lock()`（**纯函数**，条件里无任何模型名） |

### 与原方案的两处偏差（都是必要的）

1. **用两个键，而不是一个。** 原方案假设"`fixed_temperature` 单键即可"。实际 gpt-5.1 的锁是**条件性的**：仅在请求带 `reasoning_effort`（非 `none`）时才锁，reasoning 关闭时 `temperature=0` 合法。单键要么丢掉条件（把 gpt-5.1 无 reasoning 的场景也锁死 → 行为回归），要么在代码里补 `if` → 又回到原点。因此拆成 `fixed_temperature`（无条件）与 `fixed_temperature_with_reasoning`（带 reasoning 时），两者都是参数约束事实。
2. **家族表条目顺序。** `_marker_hit` 对含 `-` 的 marker 走 `marker in text`，而 `"gpt-5.1"` 包含 `"gpt-5"`；`builtin_capabilities` 是**首个命中即生效、不合并**。所以 gpt-5.1 条目必须排在 gpt-5 之前，否则永远命中不到。已在测试里用 `assert gpt51.get("fixed_temperature") is None` 锁死这个顺序。

### 不对外广告（已验证）

`capabilities_for_client_entry()` 是**显式投影**，未列出的键不输出 —— 所以 `fixed_temperature` 天然不会进入 `/v1/models`，无需为此改动客户端契约，也不会让 harness 产生新依赖。测试：`test_temperature_lock_is_never_advertised_to_clients`。

### 管理员可纠正（已验证）

- `set_model_capabilities("p1/gpt-5.6-terra", {"fixed_temperature": 0.7})` → 覆盖内置锁（`test_admin_can_override_the_builtin_lock_without_code_change`）
- 传 `null` → 清除覆盖并恢复内置启发式（`test_admin_null_restores_builtin_lock`）
- 在线注册表若声明不同值，位于内置与已存储之间，同样可覆盖

### 遗留限制（诚实记录，不要靠加 `if` 绕过）

当前**无法表达"完全不锁"**：`admin` 的 `null` 语义是"回退到内置"，不是"显式否定内置"。若某天上游彻底放开 temperature 约束，可选路径是：

1. 更新家族表条目（**一行数据**，非逻辑分支）；或
2. 由在线注册表 / admin 把锁设为客户端实际使用的值；或
3. 若确实需要"显式不锁"，应扩展 `set_model_capabilities` 的覆盖语义（例如引入 `admin_negative_keys`），**而不是在 `apply_temperature_lock` 里加模型名判断**。

### 行为等价保证

旧实现 4 个判定分支（`gpt-5` / `gpt-5-codex` / `openai/gpt-5.6` 锁；`gpt-5.1` 无 reasoning 不锁、有 reasoning 锁；其他模型不动）全部由新测试覆盖，另有能力层专项 8 个用例（normalize 校验、表一致性、不广告、admin 覆盖与恢复）。

---

## 四、登记但暂不处理

- `adapters/imagegen.py` `_is_grok_image_backend()`：`provider in {"grok", "supergrok", "xai"} or "grok-imagine" in model`
  —— 生图结果形态的厂商特例。违反 Q2，但它处在**图像生成**子系统（与本次文本链路无关），
  且当前无对应能力键可承载。建议后续与 `image_generators` 表配置合并成数据驱动。
- `core/model_capabilities.py` 的 builtin 家族表**不算特例**：它就是为"离线兜底 + 名字集中"
  设计的，且被上层 registry / upstream / admin 三层覆盖。新增模型优先加这里，不要加 `if`。
- `database.py` `_migrate_provider_options_and_headers()` 内 `"deepseek" in pid.lower()` —— 位于**一次性数据迁移**
  （`_migrate_provider_options_and_headers`，把旧 `extra_headers` 拆成
  `provider_options` / `upstream_headers`），用于保留旧部署上"deepseek 默认开 thinking"的行为。
  判定：**可接受**（不在请求路径上，不随新模型增长），但**不得在此处新增任何厂商分支**；
  迁移已完成的新部署可考虑在若干版本后清理。

### 基线快照（供 §六 第 3 条比对）

非家族表的厂商名命中，当前共 **2 处**：

```
app/adapters/imagegen.py   _is_grok_image_backend()  -> provider in {"grok", "supergrok", "xai"} or "grok-imagine" in model
app/database.py            _migrate_provider_options_and_headers() -> "deepseek" in pid.lower()  （一次性迁移内）
```

本次审计已从该基线中**移除** `services/lite_llm.py` 的 19 个视觉模型名 marker（§2.1）。
`_gpt5_temperature_supported()` 已随 §三 的收敛**删除**：`services/lite_llm.py` 现在
不再含任何模型名判断，基线维持 2 处。

---

## 五、复现本审计实验的方法

```bash
# 1. 确认 liteLLM 里 supports_vision 的真实消费点
python - <<'PY'
import litellm, inspect, pathlib
p = pathlib.Path(litellm.__file__).parent
print((p/"llms/prompt_templates/factory.py").read_text(encoding="utf-8")
      .split("def prompt_factory")[1][:4000].count("supports_vision"))
PY

# 2. 注册 vs 不注册的真实调用对比（指向死端口，看是否被本地拦截）
#    见本文 §2.1 证据 2 的描述；关键判据是错误类型是否为 "isn't mapped yet"
```


---

## 六、维护约定

1. 新增任何兼容层前，先按 AGENTS.md 四问自检，并在 PR 描述里写明归属结论。
2. 每个保留的兼容层必须满足：**有测试守护**（参考 `compatibility_patch_status()` 的做法），
   且注释里写清"替谁兜底、什么条件下可删"。
3. 定期复查本表，基线命令（只看**条件表达式**里的厂商名，排除家族表与协议枚举）：

   ```bash
   grep -rn --include=*.py -iE '"(deepseek|mimo|kimi|glm|qwen|grok|minimax|openrouter|llamacpp|pixelapi|qianye|zhipu|codex|xai)"' app/ \
     | grep -v __pycache__ | grep -v model_capabilities.py
   ```

   当前基线 2 处（见 §四快照）。命中数不应增长；新增命中必须在本文件登记理由与退出条件。
   注：家族表（`model_capabilities.py`）与 `provider_type == "anthropic"` 这类**协议枚举**
   不计入；注释里出现的厂商名属于允许的历史出处记录（如 #3、#6）——
   准则禁止的是把身份判断写进**条件表达式**。

---

## 七、外部审查复核（2026-09-22）

针对《代码改动审查报告》的逐条复核，全部以实测结论为准，不采信未复现的判断。

| 审查项 | 复核结论 | 处置 |
|---|---|---|
| **#1** webp/bmp/tiff 截断检不出 | **部分成立**。bmp/tiff 成立：实测把 BMP 截到 40%（base64 仍 1644 字符 > 门槛、仍以 `BM` 开头）会被提升为附件 —— 同款事故入口。webp 未暴露是因为另有一个 bug（见下） | ✅ 已修：`bmp` 用 `bfSize`、`webp` 用 `RIFF size` 做**容器自洽**校验；`tiff` 既无结束标记也无总长度字段 → **移出可提升集合（默认拒绝）**。未采用审查建议的"保留 + 收窄措辞"，因为那等于把已知事故入口留在原地 |
| **附带发现**（审查未提及） | `_is_webp()` 把 chunk FourCC 检查写在 `raw[8:12]`（该位置实际是 `WEBP`，`VP8 ` 在 `12:16`）→ **真实 webp 永远被拒绝**，属本次改动引入的功能 bug | ✅ 已修：`8:12` 校验 `WEBP`、`12:16` 校验 chunk 类型，并补"完整 webp 必须被提升"的用例 |
| **#2** 日志行污染（换行/控制字符） | **不成立**。`services/logger.py` 的 `JsonFormatter` 用 `json.dumps` 写日志，换行与控制字符被转义为单行 JSON 字段（实测 formatter 输出 1 行；生产 error.log 里的 traceback 也一直是单行）。再加一层转义只会变成双重转义、降低可读性 | ❌ 不转义。改为：在 `error_detail_for_log` docstring 写明该依赖；payload 单独限长（`max_chars // 2`），防止上游回整页 HTML 错误时挤掉报错文本本身 |
| **#3** `_strip_extracted` 子串前缀污染 | **成立**。base64 字符类含 `=`，已验证的短 URI 可以恰好是未验证长载荷的前缀，`str.replace` 会把长载荷前缀挖掉留碎片 | ✅ 已修：`_scan` 返回 span，删除改为**按位置从后往前**；补用例断言"文本里只剩长载荷那一个 URI 前缀" |
| **#4** 同一 URI 重复出现产生多个附件 | 成立（与旧行为一致，非回归），但 docstring 的"避免重复计费"承诺未覆盖该维度，且与预处理层"同轮重复图片去重"口径不一致 | ✅ 已修：`_scan` 内按 URI 去重 |
| **#5** bytes body 输出 `b'...'` repr | 成立 | ✅ 已修：bytes 走 `decode(errors="replace")` |
| **#6** 测试断言 `len(png_b64(2,2)) == 100` 耦合 zlib 输出长度 | 成立（脆性） | ✅ 已修：改为只断言两侧分居门槛两侧，并在注释里禁止再耦合 zlib 长度 |
| **#7** 文档行号易漂移 | 成立 | ✅ 已修：`docs/compat-patch-audit.md` 内全部改为函数名定位 |
| **仓库卫生**：`tools/scripts/_remote_exec.py` 含**明文服务器凭据** | **比"提交时留意"更严重**：该文件原先仅被 `.git/info/exclude` 忽略（本地生效、不随仓库传播），而 **SVN 侧完全没有 ignore** —— `svn status` 显示 `?`，双提交时一次 `svn add` 就会把生产 root 密码写进 SVN 历史（历史难清除） | ✅ 已把凭据类文件（`_remote_exec.py`、`.prod-server.local.md`、两份本地笔记）写入 `.gitignore`，跨机器一致防护。<br>⚠️ **仍需人工处理**：给 SVN 设置 `svn:ignore`，并把该脚本改为从 `.prod-server.local.md` 读取凭据、不再硬编码密码 |

复核后测试基线：**1020 passed**（新增 webp/bmp 自洽、tiff 拒绝、前缀污染、去重等 6 个用例）。
