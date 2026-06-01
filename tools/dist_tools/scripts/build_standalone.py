#!/usr/bin/env python
"""
LLM AIO Gateway 绿色版构建脚本

策略：
  - Python 解释器：拉 PBS install_only（带 cacert）
  - 依赖：直接把本机 venv 的 site-packages 复制到 staging/runtime/site-packages
  - 应用代码：直接复制 main.py / app/ / config.example.json / docs
  - 启动器预期目录：
        runtime/python/         嵌入式 Python
        runtime/python.tar.gz   保留 PBS 原 tarball（启动器可重新解压）
        runtime/site-packages/  离线依赖（已是"装好"状态）

打包后根目录：LLM-AIO-Gateway/

用法：
  python tools/dist_tools/scripts/build_standalone.py --target windows
  python tools/dist_tools/scripts/build_standalone.py --target all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # tools/dist_tools/scripts -> repo root
DIST_DIR = REPO_ROOT / "dist" / "standalone"
CACHE_DIR = DIST_DIR / "cache"
STAGING_DIR = DIST_DIR / "staging"
APP_SLUG = "LLM-AIO-Gateway"
PKG_ROOT = STAGING_DIR / APP_SLUG

PBS_VERSION = "20260510"
PY_VERSION = "3.12.13"
REQUIREMENTS = REPO_ROOT / "requirements.txt"

PBS_TARGETS = {
    "windows":       "x86_64-pc-windows-msvc-install_only.tar.gz",
    "macos-x86_64":  "x86_64-apple-darwin-install_only.tar.gz",
    "macos-arm64":   "aarch64-apple-darwin-install_only.tar.gz",
    "linux":         "x86_64-unknown-linux-gnu-install_only.tar.gz",
}


def banner(msg: str) -> None:
    print(f"\n{'='*60}\n  {msg}\n{'='*60}", flush=True)


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 1024 * 1024:
        print(f"  · 已缓存 {dest.name} ({dest.stat().st_size // 1024 // 1024} MB)", flush=True)
        return
    print(f"  · 下载 {url}", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, dest.open("wb") as f:
        shutil.copyfileobj(r, f)
    print(f"  · 完成 {dest.name} ({dest.stat().st_size // 1024 // 1024} MB)", flush=True)


def fetch_pbs(target: str) -> Path:
    asset = PBS_TARGETS[target]
    url = (f"https://github.com/astral-sh/python-build-standalone/releases/download/"
           f"{PBS_VERSION}/cpython-{PY_VERSION}+{PBS_VERSION}-{asset}")
    cache_sub = CACHE_DIR / "pbs"
    dest = cache_sub / asset
    download(url, dest)
    return dest


def read_version() -> str:
    pkg = REPO_ROOT / "app" / "__init__.py"
    if pkg.exists():
        text = pkg.read_text(encoding="utf-8")
        m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
        if m:
            return m.group(1)
    return time.strftime("%Y.%m.%d")


def install_dependencies_to(site: Path, pbs_python: Path) -> None:
    """Install requirements into ``site`` using the freshly extracted PBS Python.

    Works on every platform: on Windows ``pbs_python`` is
    ``runtime/python/python.exe``, on macOS / Linux it is
    ``runtime/python/bin/python3``. We invoke it as a subprocess because
    PBS does not put itself on the host sys.path.
    """
    print(f"  · 用 PBS Python 装依赖到 {site} ...", flush=True)
    cmd = [
        str(pbs_python), "-m", "pip", "install",
        "--target", str(site),
        "--no-cache",
        "--disable-pip-version-check",
        "--upgrade",
        "-r", str(REQUIREMENTS),
    ]
    print(f"    {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        sys.exit(f"依赖安装失败 (rc={proc.returncode})。检查上面的 pip 输出。")


def stage(target: str, pbs_tar: Path) -> Path:
    version = read_version()
    banner(f"拼装 staging（{target}, v{version}）")

    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    PKG_ROOT.mkdir(parents=True, exist_ok=True)
    (PKG_ROOT / "data").mkdir(exist_ok=True)
    runtime = PKG_ROOT / "runtime"
    runtime.mkdir(exist_ok=True)

    # 1. Python tarball + 解压
    print("  · 复制 Python tarball ...", flush=True)
    shutil.copy(pbs_tar, runtime / "python.tar.gz")
    print("  · 解压 Python ...", flush=True)
    with tarfile.open(pbs_tar, "r:*") as tf:
        # Python 3.14+ requires an explicit filter; pass it now to stay
        # forward-compatible and silence the DeprecationWarning.
        try:
            tf.extractall(runtime, filter="data")
        except TypeError:
            tf.extractall(runtime)
    inner = runtime / "python"
    for child in runtime.iterdir():
        if child.is_dir() and child.name == "python" and child != inner:
            if inner.exists():
                shutil.rmtree(inner)
            child.rename(inner)
            break

    # 2. 依赖（site-packages 模式）
    site = runtime / "site-packages"
    if site.exists():
        shutil.rmtree(site)
    site.mkdir()
    pbs_python = (
        runtime / "python" / ("python.exe" if target == "windows" else "bin" / "python3")
    )
    if not pbs_python.exists():
        sys.exit(f"找不到 PBS 解压后的 Python: {pbs_python}")
    install_dependencies_to(site, pbs_python)
    print(f"    site-packages 项目数: {sum(1 for _ in site.iterdir())}", flush=True)

    # 3. 应用代码
    print("  · 复制应用代码 ...", flush=True)
    for entry in ("main.py", "app", "config.example.json", "README.md",
                  "README_en.md", "使用说明书.md", "LICENSE", "docs"):
        src = REPO_ROOT / entry
        if src.is_dir():
            shutil.copytree(src, PKG_ROOT / entry,
                            ignore=shutil.ignore_patterns(
                                "__pycache__", "*.pyc", ".DS_Store"))
        elif src.is_file():
            shutil.copy(src, PKG_ROOT / entry)

    # 4. 启动器
    launcher_src = REPO_ROOT / "tools" / "dist_tools" / "launcher"
    shutil.copytree(launcher_src, PKG_ROOT / "launcher",
                    ignore=shutil.ignore_patterns("__pycache__"))

    # 5. VERSION
    (PKG_ROOT / "VERSION").write_text(version + "\n", encoding="utf-8")

    # 6. 平台入口
    _write_entrypoints(target)

    # 7. start.bat 兼容
    shutil.copy(REPO_ROOT / "start.bat", PKG_ROOT / "start.bat")
    print(f"  · staging 完成：{PKG_ROOT}", flush=True)
    return PKG_ROOT


def _write_entrypoints(target: str) -> None:
    if target.startswith("macos") or target == "linux":
        (PKG_ROOT / "run.sh").write_text(_RUN_SH, encoding="utf-8")
        (PKG_ROOT / "run.sh").chmod(0o755)

    if target == "windows":
        (PKG_ROOT / "LLM-AIO-Gateway.bat").write_text(_WIN_BAT, encoding="utf-8")
    elif target.startswith("macos"):
        app_dir = PKG_ROOT / "LLM-AIO-Gateway.app" / "Contents" / "MacOS"
        app_dir.mkdir(parents=True, exist_ok=True)
        (PKG_ROOT / "LLM-AIO-Gateway.app" / "Contents" / "Info.plist").write_text(
            _MAC_PLIST, encoding="utf-8")
        (app_dir / "LLM-AIO-Gateway").write_text(_MAC_LAUNCH, encoding="utf-8")
        (app_dir / "LLM-AIO-Gateway").chmod(0o755)
    elif target == "linux":
        (PKG_ROOT / "LLM-AIO-Gateway").write_text(_LINUX_LAUNCH, encoding="utf-8")
        (PKG_ROOT / "LLM-AIO-Gateway").chmod(0o755)


# 入口：直接调 PBS 嵌入式 Python，不依赖系统 Python
_RUN_SH = r"""#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/runtime/python/bin/python3" "$DIR/launcher/gui.py" "$@"
"""

_WIN_BAT = r"""@echo off
setlocal
cd /d "%~dp0"
start "" "%~dp0runtime\python\pythonw.exe" "%~dp0launcher\gui.py"
"""

_LINUX_LAUNCH = r"""#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/runtime/python/bin/python3" "$DIR/launcher/gui.py" "$@"
"""

_MAC_LAUNCH = r"""#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")/../.." && pwd)"
exec "$DIR/runtime/python/bin/python3" "$DIR/launcher/gui.py"
"""

_MAC_PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>LLM-AIO-Gateway</string>
  <key>CFBundleDisplayName</key><string>LLM AIO Gateway</string>
  <key>CFBundleIdentifier</key><string>com.llm-aio.gateway</string>
  <key>CFBundleVersion</key><string>1.0.0</string>
  <key>CFBundleExecutable</key><string>LLM-AIO-Gateway</string>
  <key>LSUIElement</key><false/>
</dict>
</plist>
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pack(target: str, version: str) -> Path:
    banner(f"压缩 {target}")
    if target == "windows" or target.startswith("macos"):
        out = DIST_DIR / f"{APP_SLUG}-{target}-v{version}.zip"
        if out.exists():
            out.unlink()
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6) as zf:
            for f in PKG_ROOT.rglob("*"):
                if f.is_file():
                    arc = f"{APP_SLUG}/" + f.relative_to(PKG_ROOT).as_posix()
                    zf.write(f, arc)
    elif target == "linux":
        out = DIST_DIR / f"{APP_SLUG}-{target}-v{version}.tar.gz"
        if out.exists():
            out.unlink()
        with tarfile.open(out, "w:gz") as tf:
            for f in PKG_ROOT.rglob("*"):
                if f.is_file():
                    arc = f"{APP_SLUG}/" + f.relative_to(PKG_ROOT).as_posix()
                    tf.add(f, arc)
    else:
        sys.exit(f"未知 target: {target}")

    size_mb = out.stat().st_size // 1024 // 1024
    print(f"  · {out.name}  {size_mb} MB  sha256={sha256(out)[:12]}...", flush=True)
    return out


def write_version_json(version: str, artifacts: list[tuple[str, Path]]) -> None:
    out = DIST_DIR / "version.json"
    payload = {
        "app": APP_SLUG,
        "version": version,
        "channel": "stable",
        "released_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "notes": "绿色版发布",
        "artifacts": [
            {"target": t, "filename": p.name,
             "size": p.stat().st_size, "sha256": sha256(p)}
            for t, p in artifacts
        ],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"  · {out}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="windows",
                        choices=["windows", "macos-x86_64", "macos-arm64",
                                 "linux", "all"])
    args = parser.parse_args()

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    targets = (["windows", "macos-x86_64", "macos-arm64", "linux"]
               if args.target == "all" else [args.target])

    version = read_version()
    artifacts: list[tuple[str, Path]] = []

    for t in targets:
        banner(f"构建 {t}")
        pbs = fetch_pbs(t)
        stage(t, pbs)
        out = pack(t, version)
        artifacts.append((t, out))

    write_version_json(version, artifacts)
    banner("✅ 完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
