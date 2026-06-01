"""
LLM AIO Gateway 绿色版启动器（双击即用）

- 首次启动：自动解压嵌入式 Python
- 日常启动：site-packages 是构建时已复制好的，直接 import 验证即可
- 按钮：启动 / 停止 / 打开管理面板 / 打开数据目录 / 检查更新
- 跨平台：Windows / macOS / Linux 共用同一份 Tkinter 代码
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from tkinter import (
    END,
    StringVar,
    Tk,
    messagebox,
    scrolledtext,
    ttk,
)

APP_NAME = "LLM AIO Gateway"
APP_SLUG = "LLM-AIO-Gateway"

# 启动器位于 <ROOT_DIR>/launcher/gui.py
ROOT_DIR = Path(__file__).resolve().parent.parent
PYTHON_DIR = ROOT_DIR / "runtime" / "python"
SITE_DIR = ROOT_DIR / "runtime" / "site-packages"
DATA_DIR = ROOT_DIR / "data"
CONFIG_FILE = DATA_DIR / "config.json"
VENV_MARKER = PYTHON_DIR / "aio_installed.marker"
LOG_FILE = ROOT_DIR / "launcher.log"
UPDATE_URL = os.environ.get(
    "LLM_AIO_UPDATE_URL",
    "https://raw.githubusercontent.com/your-org/llm-aio-gateway/main/dist/standalone/version.json",
)

DEFAULT_PORT = 8000

IS_WINDOWS = sys.platform.startswith("win")
IS_MAC = sys.platform == "darwin"
PY_EXE = (
    PYTHON_DIR / "python.exe" if IS_WINDOWS else PYTHON_DIR / "bin" / "python3"
)

CHILD_ENV_BASE = os.environ.copy()


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def log(msg: str) -> str:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    return line


def http_get(url: str, timeout: float = 5.0) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read()
    except (urllib.error.URLError, socket.timeout, ConnectionError) as exc:
        log(f"GET {url} failed: {exc}")
        return None


def free_port_hint(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        try:
            s.connect(("127.0.0.1", port))
            return False
        except OSError:
            return True


# --------------------------------------------------------------------------- #
# 首次启动：解压 Python（site-packages 由构建时复制，无需安装）
# --------------------------------------------------------------------------- #
def ensure_python_and_deps(progress) -> bool:
    """确保 Python + site-packages 都可用。percent=-1 表示错误。"""
    if VENV_MARKER.exists() and PY_EXE.exists() and SITE_DIR.exists():
        progress(100, "运行时已就绪")
        return True

    if not PY_EXE.exists():
        progress(10, "正在解压嵌入式 Python ...")
        if not _extract_python(progress):
            return False

    if not SITE_DIR.exists() or not any(SITE_DIR.iterdir()):
        progress(-1, f"找不到依赖目录：{SITE_DIR}（构建产物不完整，请重新下载）")
        return False

    # site-packages 是构建时复制好的"已装好"的目录。
    # 在 PBS 嵌入式 Python 下不会自动加入 sys.path，所以我们用子进程验证。
    progress(60, "验证依赖 ...")
    if not _smoke_test():
        progress(-1, "依赖验证失败：fastapi/litellm/uvicorn 至少一个无法 import（看 launcher.log）")
        return False

    VENV_MARKER.write_text(
        f"installed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8"
    )
    progress(100, "已就绪")
    return True


def _find_python_tarball() -> Path | None:
    for c in (ROOT_DIR / "runtime" / "python.tar.gz",
              ROOT_DIR / "runtime" / "python.zip"):
        if c.exists():
            return c
    return None


def _extract_python(progress) -> bool:
    archive = _find_python_tarball()
    if not archive:
        progress(-1, "找不到嵌入式 Python 包：runtime/python.tar.gz")
        return False
    PYTHON_DIR.parent.mkdir(parents=True, exist_ok=True)
    try:
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(PYTHON_DIR.parent)
        else:
            with tarfile.open(archive, "r:*") as tf:
                tf.extractall(PYTHON_DIR.parent)
    except (tarfile.TarError, zipfile.BadZipFile, OSError) as exc:
        log(f"extract python failed: {exc}")
        return False

    inner = PYTHON_DIR.parent / "python"
    if not PYTHON_DIR.exists() and inner.exists():
        inner.rename(PYTHON_DIR)
    else:
        for child in PYTHON_DIR.parent.iterdir():
            if child.is_dir() and child.name == "python" and child != PYTHON_DIR:
                if PYTHON_DIR.exists():
                    shutil.rmtree(PYTHON_DIR)
                child.rename(PYTHON_DIR)
                break

    if not PY_EXE.exists():
        log(f"python executable not found at {PY_EXE}")
        return False
    return True


def _smoke_test() -> bool:
    """用子进程调 PBS python + PYTHONPATH 指向 site-packages，验证关键依赖能 import"""
    cmd = [str(PY_EXE), "-c",
           "import fastapi, litellm, pydantic, uvicorn, httpx, anyio; print('OK')"]
    env = _child_env()
    try:
        out = subprocess.check_output(
            cmd, env=env, stderr=subprocess.STDOUT, timeout=30)
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired) as exc:
        # 把 stderr 写到日志，便于排查
        if isinstance(exc, subprocess.CalledProcessError):
            log("smoke test stderr:\n" + exc.output.decode("utf-8", "replace"))
        else:
            log(f"smoke test failed: {exc}")
        return False
    return b"OK" in out


def _child_env() -> dict:
    env = CHILD_ENV_BASE.copy()
    # PBS 嵌入式 Python 不会自动加载 site-packages，必须显式塞进 PYTHONPATH
    pp = str(SITE_DIR)
    env["PYTHONPATH"] = pp + os.pathsep + env.get("PYTHONPATH", "")
    if IS_MAC:
        env.setdefault("LANG", "en_US.UTF-8")
    return env


# --------------------------------------------------------------------------- #
# 启动 / 停止后端
# --------------------------------------------------------------------------- #
class Backend:
    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.port = DEFAULT_PORT
        self._configure()

    def _configure(self) -> None:
        if CONFIG_FILE.exists():
            try:
                cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                self.port = int(cfg.get("port", DEFAULT_PORT))
            except (OSError, ValueError):
                pass

    def is_running(self) -> bool:
        with self.lock:
            if self.proc and self.proc.poll() is None:
                return True
            return False

    def start(self) -> tuple[bool, str]:
        with self.lock:
            if self.proc and self.proc.poll() is None:
                return False, "后端已经在运行"
            if not free_port_hint(self.port):
                return False, f"端口 {self.port} 已被占用"

            DATA_DIR.mkdir(parents=True, exist_ok=True)
            log_path = DATA_DIR / "logs" / "backend.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("ab", buffering=0)

            env = _child_env()
            env["PYTHONUNBUFFERED"] = "1"
            cmd = [str(PY_EXE), "-u", "-m", "uvicorn", "main:app",
                   "--host", "0.0.0.0", "--port", str(self.port)]
            log("launch: " + " ".join(cmd))
            try:
                self.proc = subprocess.Popen(
                    cmd, cwd=ROOT_DIR, env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log_file, stderr=log_file,
                    creationflags=(subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0),
                )
            except OSError as exc:
                return False, f"启动失败: {exc}"

            for _ in range(40):
                time.sleep(0.25)
                if not free_port_hint(self.port):
                    return True, f"后端已启动 (PID {self.proc.pid})"
                if self.proc.poll() is not None:
                    return False, "后端进程已退出，详情见 logs/backend.log"
            return False, "等待端口超时"

    def stop(self) -> tuple[bool, str]:
        with self.lock:
            if not self.proc or self.proc.poll() is not None:
                self.proc = None
                return False, "后端未在运行"
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=3)
            self.proc = None
            return True, "后端已停止"


# --------------------------------------------------------------------------- #
# 更新检查
# --------------------------------------------------------------------------- #
def check_update(current_version: str) -> tuple[bool, str, str]:
    body = http_get(UPDATE_URL, timeout=5.0)
    if not body:
        return False, "", "无法访问更新服务器"
    try:
        info = json.loads(body.decode("utf-8-sig"))
    except json.JSONDecodeError as exc:
        return False, "", f"更新文件解析失败: {exc}"
    latest = info.get("version", "")
    notes = info.get("notes", "")
    if latest and latest != current_version:
        return True, latest, notes or "有可用更新"
    return False, latest or current_version, "已是最新版本"


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
class App:
    def __init__(self) -> None:
        self.root = Tk()
        self.root.title(APP_NAME)
        self.root.geometry("720x520")
        self.root.minsize(640, 460)
        self._setup_style()

        self.backend = Backend()
        self.current_version = self._read_version()
        self.log_q: queue.Queue = queue.Queue()
        self.bootstrap_done = False

        self._build_ui()
        self._poll_log()
        self._poll_status()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.root.after(200, self._auto_bootstrap)

    def _read_version(self) -> str:
        f = ROOT_DIR / "VERSION"
        if f.exists():
            return f.read_text(encoding="utf-8").strip() or "dev"
        return "dev"

    def _setup_style(self) -> None:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "aqua" in style.theme_names():
            style.theme_use("aqua")
        else:
            style.theme_use(style.theme_names()[0])
        style.configure("Big.TButton", font=("Segoe UI", 11, "bold"), padding=8)
        style.configure("Status.TLabel", padding=(8, 4))

    def _build_ui(self) -> None:
        header = ttk.Frame(self.root, padding=12)
        header.pack(fill="x")
        ttk.Label(header, text=APP_NAME, font=("Segoe UI", 16, "bold")).pack(side="left")
        ttk.Label(header, text=f"  v{self.current_version}").pack(side="left")

        body = ttk.Frame(self.root, padding=(12, 0))
        body.pack(fill="both", expand=True)

        status_box = ttk.LabelFrame(body, text="运行状态", padding=12)
        status_box.pack(fill="x")

        self.status_var = StringVar(value="未启动")
        self.dot_var = StringVar(value="●")
        self.dot_color = "red"
        ttk.Label(status_box, textvariable=self.dot_var, foreground=self.dot_color,
                  font=("Segoe UI", 18, "bold")).grid(row=0, column=0, padx=(0, 8))
        ttk.Label(status_box, textvariable=self.status_var,
                  font=("Segoe UI", 12)).grid(row=0, column=1, sticky="w")

        info_line = ttk.Frame(status_box)
        info_line.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.url_var = StringVar(value=f"http://localhost:{self.backend.port}/")
        ttk.Label(info_line, textvariable=self.url_var,
                  foreground="#0066cc").pack(side="left")
        ttk.Button(info_line, text="复制", width=5,
                   command=lambda: self._copy(self.url_var.get())).pack(side="left", padx=4)

        actions = ttk.Frame(body, padding=(0, 12))
        actions.pack(fill="x")
        ttk.Button(actions, text="▶  启动服务", style="Big.TButton",
                   command=self.on_start).pack(side="left", padx=4)
        ttk.Button(actions, text="■  停止服务", style="Big.TButton",
                   command=self.on_stop).pack(side="left", padx=4)
        ttk.Button(actions, text="打开管理面板",
                   command=self.on_open_panel).pack(side="left", padx=4)
        ttk.Button(actions, text="打开数据目录",
                   command=self.on_open_data).pack(side="left", padx=4)
        ttk.Button(actions, text="检查更新",
                   command=self.on_check_update).pack(side="right", padx=4)

        log_box = ttk.LabelFrame(body, text="日志", padding=8)
        log_box.pack(fill="both", expand=True)
        self.log_widget = scrolledtext.ScrolledText(
            log_box, height=12, state="disabled", font=("Consolas", 9),
            background="#1e1e1e", foreground="#d4d4d4", insertbackground="white",
        )
        self.log_widget.pack(fill="both", expand=True)

        self.progress_var = StringVar(value="就绪")
        footer = ttk.Frame(self.root, padding=(12, 4))
        footer.pack(fill="x")
        ttk.Label(footer, textvariable=self.progress_var,
                  style="Status.TLabel").pack(side="left")
        ttk.Label(footer, text=f"安装目录：{ROOT_DIR}",
                  style="Status.TLabel").pack(side="right")

    def _copy(self, text: str) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        messagebox.showinfo(APP_NAME, "已复制到剪贴板")

    def on_start(self) -> None:
        if not self.bootstrap_done:
            messagebox.showinfo(APP_NAME, "正在完成首次安装，请稍候 ...")
            return
        ok, msg = self.backend.start()
        (messagebox.showinfo if ok else messagebox.showerror)(APP_NAME, msg)
        self._append_log(msg)

    def on_stop(self) -> None:
        ok, msg = self.backend.stop()
        (messagebox.showinfo if ok else messagebox.showwarning)(APP_NAME, msg)
        self._append_log(msg)

    def on_open_panel(self) -> None:
        import webbrowser
        webbrowser.open(self.url_var.get())

    def on_open_data(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._reveal(DATA_DIR)

    def on_check_update(self) -> None:
        def worker():
            has, ver, notes = check_update(self.current_version)
            self.root.after(0, lambda: messagebox.showinfo(
                APP_NAME, f"当前版本：v{self.current_version}\n最新版本：v{ver}\n\n{notes}"))
        threading.Thread(target=worker, daemon=True).start()

    def _reveal(self, path: Path) -> None:
        if IS_MAC:
            subprocess.Popen(["open", str(path)])
        elif IS_WINDOWS:
            subprocess.Popen(["explorer", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])

    def _auto_bootstrap(self) -> None:
        if self.bootstrap_done:
            return
        def worker():
            def progress(percent, text):
                self.log_q.put(("BOOT", percent, text))
            ok = ensure_python_and_deps(progress)
            self.log_q.put(("BOOT_DONE", 100 if ok else -1,
                            "就绪" if ok else "首次安装失败"))
        threading.Thread(target=worker, daemon=True).start()

    def _poll_log(self) -> None:
        try:
            while True:
                item = self.log_q.get_nowait()
                if item[0] == "BOOT":
                    _, percent, text = item
                    self.progress_var.set(text)
                    self._append_log(text)
                elif item[0] == "BOOT_DONE":
                    _, percent, text = item
                    self.bootstrap_done = percent == 100
                    self.progress_var.set(text)
                    self._append_log(text)
        except queue.Empty:
            pass
        self.root.after(150, self._poll_log)

    def _poll_status(self) -> None:
        if self.backend.is_running():
            self.status_var.set(f"运行中 · 端口 {self.backend.port}")
            self.dot_var.set("●")
            self.dot_color = "#2ea44f"
        else:
            self.status_var.set("未启动")
            self.dot_var.set("●")
            self.dot_color = "#cc3344"
        self.root.after(600, self._poll_status)

    def _append_log(self, text: str) -> None:
        self.log_widget.configure(state="normal")
        self.log_widget.insert(END, log(text) + "\n")
        self.log_widget.see(END)
        self.log_widget.configure(state="disabled")

    def _on_close(self) -> None:
        if self.backend.is_running():
            if not messagebox.askyesno(APP_NAME, "服务还在运行，确定要关闭吗？"):
                return
            self.backend.stop()
        self.root.destroy()


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        app = App()
        app.root.mainloop()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        try:
            messagebox.showerror(APP_NAME, f"启动器异常：{exc}")
        except Exception:
            pass
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
