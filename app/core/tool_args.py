import json
from typing import Any

# 目标词用拼接构造，避免本文件被任何按字面量过滤的工具误识别（与 tool_leak 同惯例）。
_UNDEFINED = "und" + "efined"

# 字符串边界扫描时，闭引号只有在前面有奇数个反斜杠时才是被转义的。
# 旧实现只看 args[i-1] != '\\'，把 "x\\"（值以反斜杠结尾）的闭引号误判为
# 转义引号，in_str 从此失步，导致引号外的目标词永远修不掉（bug-2026-10-05 M-2）。


def _trailing_backslashes(text: str, end: int) -> int:
    count = 0
    i = end - 1
    while i >= 0 and text[i] == "\\":
        count += 1
        i -= 1
    return count


def sanitize_args(args: str) -> str:
    """把 JSON 文本中字符串外的裸目标词替换为 ``""``（整串一次扫描）。

    字符串跟踪按转义奇偶判定；这是非流式路径（fix_tool_args）的实现。
    """
    out: list[str] = []
    in_str = False
    i = 0
    n = len(args)
    while i < n:
        c = args[i]
        if c == '"':
            if not _trailing_backslashes(args, i) % 2:
                in_str = not in_str
            out.append(c)
            i += 1
            continue
        if not in_str and args[i:i + len(_UNDEFINED)] == _UNDEFINED:
            end = i + len(_UNDEFINED)
            if end >= n or args[end] in ',}] \t\r\n:':
                out.append('""')
                i = end
                continue
        out.append(c)
        i += 1
    return ''.join(out)


class StreamingArgsSanitizer:
    """跨分片增量替换字符串外的裸目标词（bug-2026-10-05 M-1）。

    旧实现对每个分片单独跑无状态扫描：字符串跟踪在每个分片都从头开始，
    字符串值内合法的裸词文本会被误改；且分片级与整体级两次扫描轨迹不同，
    客户端按 delta 重组的参数与落库的最终参数可能不一致。

    本类把字符串状态与"可能是目标词前缀"的尾部字节跨分片保持，每个输入
    字节只被处理一次：feed() 返回可安全下发的已修复前缀，flush() 吐出
    残留。分片序列经 feed/flush 的输出等价于对整串做一次 sanitize_args。
    """

    __slots__ = ("_in_str", "_pending")

    def __init__(self) -> None:
        self._in_str = False
        self._pending = ""

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        buf = self._pending + chunk
        self._pending = ""
        out: list[str] = []
        i = 0
        n = len(buf)
        while i < n:
            c = buf[i]
            if self._in_str:
                if c == '"' and not _trailing_backslashes(buf, i) % 2:
                    self._in_str = False
                out.append(c)
                i += 1
                continue
            if c == '"':
                self._in_str = True  # 字符串外的引号即开引号（前面无转义语境）
                out.append(c)
                i += 1
                continue
            if c == _UNDEFINED[0]:
                candidate = buf[i:i + len(_UNDEFINED)]
                # 尾部不完整：可能是跨分片的目标词前缀，扣住等待下一个分片。
                if len(candidate) < len(_UNDEFINED) and _UNDEFINED.startswith(candidate):
                    self._pending = buf[i:]
                    break
                if candidate == _UNDEFINED:
                    end = i + len(_UNDEFINED)
                    next_char = buf[end] if end < n else None
                    if next_char is None:
                        # 词完整但边界未知（分片恰好切在词后）：扣住整词。
                        self._pending = buf[i:]
                        break
                    if next_char in ',}] \t\r\n:':
                        out.append('""')
                        i = end
                        continue
            out.append(c)
            i += 1
        if not self._pending and self._in_str:
            # 闭合引号的转义奇偶依赖引号前的反斜杠串；串尾的反斜杠若已放行，
            # 下一分片的引号就会数错奇偶。把尾串扣到下一分片一起处理。
            tail = 0
            while n - tail - 1 >= 0 and buf[n - tail - 1] == "\\":
                tail += 1
            if tail:
                self._pending = buf[n - tail:]
                del out[-tail:]
        return ''.join(out)

    def flush(self) -> str:
        pending, self._pending = self._pending, ""
        if not pending:
            return ""
        # 残留按整串口径处理：字符串内（含悬挂反斜杠）原样输出；字符串外
        # 若恰是完整目标词则替换（流尾即边界）。
        if self._in_str:
            return pending
        if pending == _UNDEFINED:
            return '""'
        return sanitize_args(pending)


def coerce_tool_arguments_json(raw: Any) -> str:
    """Return tool-call arguments as a JSON object string.

    llama.cpp rejects historical tool calls whose arguments are not a JSON
    object. Keep already-valid objects unchanged; wrap anything else.
    """
    if raw is None:
        return "{}"
    if not isinstance(raw, str):
        try:
            raw = json.dumps(raw, ensure_ascii=False)
        except (TypeError, ValueError):
            return "{}"
    if not raw.strip():
        return "{}"
    try:
        parsed = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (json.JSONDecodeError, ValueError):
        return json.dumps({"input": raw}, ensure_ascii=False)
    if isinstance(parsed, dict):
        return raw
    return json.dumps({"value": parsed}, ensure_ascii=False)


def fix_tool_args(tc_dict: dict) -> None:
    func = tc_dict.get("function")
    if not func or not isinstance(func, dict):
        return
    args = func.get("arguments", "")
    if args and _UNDEFINED in args:
        func["arguments"] = sanitize_args(args)
