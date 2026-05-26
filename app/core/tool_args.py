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


def fix_tool_args(tc_dict: dict) -> None:
    func = tc_dict.get("function")
    if not func or not isinstance(func, dict):
        return
    args = func.get("arguments", "")
    if args and "undefined" in args:
        func["arguments"] = sanitize_args(args)
