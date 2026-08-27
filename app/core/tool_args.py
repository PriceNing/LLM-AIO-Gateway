import json
from typing import Any


def sanitize_args(args: str) -> str:
    out = []
    in_str = False
    i = 0
    n = len(args)
    while i < n:
        c = args[i]
        if c == '"' and (i == 0 or args[i - 1] != '\\'):
            in_str = not in_str
        if not in_str and args[i:i + 9] == 'undefined':
            end = i + 9
            if end >= n or args[end] in ',}]\n\r\t ':
                out.append('""')
                i = end
                continue
        out.append(c)
        i += 1
    return ''.join(out)


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
    if args and "undefined" in args:
        func["arguments"] = sanitize_args(args)
