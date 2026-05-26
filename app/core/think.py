def extract_and_strip_think(text: str) -> tuple[str, str]:
    if not text:
        return text, ""
    think_parts = []
    result = []
    i = 0
    while i < len(text):
        start = text.find("<think>", i)
        if start == -1:
            result.append(text[i:])
            break
        result.append(text[i:start])
        depth = 1
        pos = start + 7
        while depth > 0 and pos < len(text):
            next_open = text.find("<think>", pos)
            next_close = text.find("</think>", pos)
            if next_close == -1:
                pos = -1
                break
            if next_open != -1 and next_open < next_close:
                depth += 1
                pos = next_open + 7
            else:
                depth -= 1
                if depth == 0:
                    think_parts.append(text[start + 7:next_close])
                pos = next_close + 8
        if pos == -1:
            result.append(text[start:])
            break
        while pos < len(text) and text[pos] in " \t\n\r\f":
            pos += 1
        i = pos
    return "".join(result).strip(), "\n".join(think_parts)


def strip_think_tags(text: str) -> str:
    cleaned, _ = extract_and_strip_think(text)
    return cleaned
