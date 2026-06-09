# LLM AIO Gateway 绿色版（零安装发布包）

面向非程序员用户的"双击即用"发布方式。把 Python 解释器、全部依赖和应用代码打包成单一目录（Windows 上是 `.zip`、macOS 是 `.app` + zip、Linux 是 `.tar.gz`），用户解压后双击即可启动 GUI 启动器，启动器会自动解压 Python、装依赖、跑服务。

## 目录结构

```
tools/standalone/
├── README.md                   # 本文件
├── launcher/
│   └── gui.py                  # Tkinter 启动器（启动/停止/打开面板/检查更新）
└── scripts/
    └── build_standalone.py     # 构建脚本
```

## 给开发者：如何打一个绿色版

### 一次性准备

```bash
# 安装 uv（构建依赖用的工具）
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
# Windows 用 PowerShell: irm https://astral.sh/uv/install.ps1 | iex
```

### 跑构建

```bash
# Windows
python tools/standalone/scripts/build_standalone.py --target windows

# macOS（x86_64 / arm64 分开打，发布时同时给两个）
python tools/standalone/scripts/build_standalone.py --target macos-arm64
python tools/standalone/scripts/build_standalone.py --target macos-x86_64

# Linux
python tools/standalone/scripts/build_standalone.py --target linux

# 或者一次打完所有平台（仅当前机器能打的）
python tools/standalone/scripts/build_standalone.py --target all
```

### 输出

```
dist/standalone/
├── LLM-AIO-Gateway-windows-x86_64-v0.3.1.zip
├── LLM-AIO-Gateway-macos-arm64-v0.3.1.zip
├── LLM-AIO-Gateway-macos-x86_64-v0.3.1.zip
├── LLM-AIO-Gateway-linux-x86_64-v0.3.1.tar.gz
├── version.json
└── cache/                       # wheels + PBS，重复构建会复用
```

## 用户视角：解压后长什么样

```
LLM-AIO-Gateway/
├── LLM-AIO-Gateway.app/                 # macOS 用户双击这个
├── LLM-AIO-Gateway                      # Linux 入口（绿色图标）
├── LLM-AIO-Gateway.bat / .vbs           # Windows 入口（双击 LLM-AIO-Gateway.bat）
├── start.bat                    # 兼容旧 start.bat
├── launcher/gui.py              # 启动器
├── app/                         # 应用代码
├── main.py
├── config.example.json
├── runtime/
│   ├── python.tar.gz            # 嵌入式 Python（按平台）
│   └── wheels/                  # 离线依赖
├── data/                        # 首次启动后自动生成
└── VERSION
```

## 启动流程（用户视角）

1. 双击 `LLM-AIO-Gateway.bat`（Windows）/ 双击 `LLM-AIO-Gateway`（Linux）/ 双击 `LLM-AIO-Gateway.app`（macOS）
2. 弹出 GUI 启动器
3. 第一次启动会做"首次安装"：
   - 解压 `runtime/python.tar.gz` 到 `runtime/python/`
   - 用嵌入式 Python + 离线 wheels 装依赖（`pip install --no-index -f runtime/wheels`）
   - 标记 `runtime/python/aio_installed.marker`（下次直接跳过）
4. 点 **▶ 启动服务** 按钮，后台 `uvicorn main:app` 起在 8000 端口
5. 点 **打开管理面板** 浏览器跳到 `http://localhost:8000`
6. 点 **检查更新** 会拉 `version.json` 比对版本

## 自更新

启动器从环境变量 `LLM_AIO_UPDATE_URL` 拉 `version.json`：

```json
{
  "version": "0.3.1",
  "released_at": "2026-06-15T00:00:00Z",
  "notes": "修复 xxx",
  "artifacts": [
    {"target": "windows", "filename": "LLM-AIO-Gateway-windows-x86_64-v0.3.1.zip",
     "size": 188743424, "sha256": "abc..."},
    ...
  ]
}
```

启动器只做"提示 + 打开下载页"，不自动替换（避免 Windows 上文件占用）。后续可以扩展成"下载到新目录 + 提示用户迁移 data/"。

## 设计决策

- **为什么不用 PyInstaller？** 体积爆炸（一个 hello world 也要 30MB+，加 litellm 之后超过 500MB），冷启动慢，而且 `litellm` 的动态 import 跟 PyInstaller 兼容性差。
- **为什么选 PBS（python-build-standalone）？** 25MB 自带 pip + SSL + tkinter，零依赖，跨平台。解压就能用。
- **为什么用 uv 解析依赖？** 比 pip 快 10-100 倍，依赖解析更稳。
- **为什么依赖也打包成离线 wheel？** 用户机器上可能没有外网。
- **为什么不直接 PyInstaller 打包整个 `gui.py`？** 跨平台、跨 Python 版本简单；启动器升级只需替换 `launcher/gui.py`，不影响应用代码。

## 已知限制

- 启动器依赖系统 Tk（Windows / macOS 自带；Linux 需要 `apt install python3-tk`）
- 启动器调用 `pythonw.exe`（Windows）/ `python3`（macOS / Linux）；如果用户有多个 Python 建议在 README 里强调
- macOS 首次双击 `.app` 可能被 Gatekeeper 拦截，需要在"系统设置 → 隐私与安全"中点"仍要打开"
- 体积约 200-300MB（litellm 占大头）；可以考虑加 `--without-deps` 后按需补装

## 调试

启动器日志在 `LLM-AIO-Gateway/launcher.log`，后端日志在 `LLM-AIO-Gateway/data/logs/backend.log`。
