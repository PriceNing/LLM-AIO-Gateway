def extract_and_strip_think(text: str) -> tuple[str, str]:
    """抽取并剥离 <think>...</think> 块，返回 (正文, reasoning)。

    边界口径（bug-2026-10-05 L-24，与流式路径 iter_openai_chat_output_events
    收尾处理一致）：只有开标签、到文本结束仍未闭合的 <think> 块按 reasoning
    处理（开标签本身一并剥离），不再原样留在正文——正文泄漏未闭合思考过程
    比截断更糟，且两条路径口径必须一致。嵌套 <think> 需配对闭合。
    """
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
            # 未闭合：标签后的全部内容归 reasoning（对齐流式收尾口径）。
            think_parts.append(text[start + 7:])
            break
        while pos < len(text) and text[pos] in " \t\n\r\f":
            pos += 1
        i = pos
    return "".join(result).strip(), "\n".join(think_parts)


def strip_think_tags(text: str) -> str:
    cleaned, _ = extract_and_strip_think(text)
    return cleaned
