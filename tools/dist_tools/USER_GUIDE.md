# LLM AIO Gateway 绿色版使用说明

> 不用装 Python，不用装 Docker，**解压双击就能用**。

## 1. 下载

从 [GitHub Releases](https://github.com/pricening/llm-aio-gateway/releases) 找到最新版本，下载对应你电脑的压缩包：

| 你的电脑 | 下载 |
|---|---|
| Windows 10/11 | `LLM-AIO-Gateway-windows-x86_64-vX.Y.Z.zip` |
| macOS Apple Silicon（M1/M2/M3/M4） | `LLM-AIO-Gateway-macos-arm64-vX.Y.Z.zip` |
| Linux | `LLM-AIO-Gateway-linux-x86_64-vX.Y.Z.tar.gz` |

## 2. 解压

把压缩包解压到任意位置，**路径里不要有中文**（建议 `C:\LLM-AIO-Gateway` 或 `~/LLM-AIO-Gateway`）。

解压后会得到一个 `LLM-AIO-Gateway` 文件夹。

## 3. 启动

| 系统 | 操作 |
|---|---|
| **Windows** | 打开 `LLM-AIO-Gateway` 文件夹 → 双击 `LLM-AIO-Gateway.bat` |
| **macOS** | 打开 `LLM-AIO-Gateway` 文件夹 → 双击 `LLM-AIO-Gateway.app`<br>（首次会被拦截，到"系统设置 → 隐私与安全性"点"仍要打开"） |
| **Linux** | 终端里 `cd LLM-AIO-Gateway && ./LLM-AIO-Gateway` |

> 第一次启动会先做"安装"：解压 Python + 装依赖，需要 1-3 分钟，**期间不要关窗口**。等出现绿色 ● 和"运行中"字样，就完成了。

## 4. 使用

启动后会自动打开管理面板（如果没自动开，手动访问 `http://localhost:8000`）。

- 第一次使用需要 **创建管理员账号**
- 在管理面板添加 **上游提供商**（OpenAI 兼容 / Anthropic 兼容 / 自建服务都行）
- 创建 **API Key**，把它填到 Claude Code / Codex / OpenCode 等客户端里，base URL 设为 `http://localhost:8000/v1` 即可

## 5. 日常使用

- 启动器可以最小化到任务栏（Windows）/ 菜单栏（macOS）
- 不再使用时点 **■ 停止服务** 再关闭
- 所有用户数据（配置、数据库、日志）都在 `LLM-AIO-Gateway/data/`，重装时**只要保留这个文件夹就不会丢数据**

## 6. 出问题怎么办

- **服务起不来** → 看启动器窗口的日志区，常见原因是端口被占
- **想换个端口** → 编辑 `data/config.json`，把 `port` 改成别的（比如 8800），重启
- **想完全重置** → 删掉 `data/` 文件夹再启动

## 7. 更新

启动器会定期检查更新（也可点"检查更新"按钮）。提示有新版本时：

1. 下载新的压缩包
2. 解压到一个**新文件夹**（不要覆盖！）
3. 把旧 `LLM-AIO-Gateway/data/` 复制到新文件夹
4. 双击新文件夹的启动器

> 不要把新版本解压到旧文件夹里面再启动——会导致两个版本的 `data/` 混在一起。

---

## 卸载

直接删除 `LLM-AIO-Gateway` 文件夹即可，没有任何注册表 / 系统服务残留。
