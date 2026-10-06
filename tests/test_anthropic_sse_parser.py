import pytest

from app.adapters.anthropic_streaming import _iter_anthropic_sse_events


@pytest.mark.asyncio
async def test_sse_parser_accepts_compact_fields_and_multiline_data():
    """bug-2026-10-05 L-3：紧凑 `data:{...}`（冒号后无空格）与多行 data 拼接。"""

    async def lines():
        yield "event:message_start"
        yield 'data:{"type":"message_start","message":{"usage":{"input_tokens":1}}}'
        yield ""
        yield "event:content_block_delta"
        yield 'data:{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"he'
        yield 'data:llo"}}'
        yield ""

    got = [item async for item in _iter_anthropic_sse_events(lines())]
    assert got == [
        ("message_start", '{"type":"message_start","message":{"usage":{"input_tokens":1}}}'),
        ("content_block_delta",
         '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"he'
         + "\n"
         + 'llo"}}'),
    ]


@pytest.mark.asyncio
async def test_sse_parser_dispatches_without_blank_line_on_event_boundary():
    """部分实现不发空行：新 event 字段到达即派发上一事件（旧逐行行为的等价物）。"""

    async def lines():
        yield "event: content_block_delta"
        yield 'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"a"}}'
        yield "event: content_block_stop"
        yield 'data: {"type":"content_block_stop","index":0}'

    got = [item async for item in _iter_anthropic_sse_events(lines())]
    assert len(got) == 2
    assert got[0][0] == "content_block_delta"
    assert got[1][0] == "content_block_stop"


@pytest.mark.asyncio
async def test_sse_parser_flushes_buffered_event_before_propagating_error():
    """异常断流时已缓冲的完整 data 必须先派发，不得吞掉已收到的内容。"""

    async def lines():
        yield "event: content_block_delta"
        yield 'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"x"}}'
        raise RuntimeError("upstream died")

    collected = []
    with pytest.raises(RuntimeError):
        async for item in _iter_anthropic_sse_events(lines()):
            collected.append(item)
    assert collected == [
        ("content_block_delta",
         '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"x"}}'),
    ]


@pytest.mark.asyncio
async def test_sse_parser_stops_at_done_sentinel():
    async def lines():
        yield 'data: {"type":"ping"}'
        yield ""
        yield "data: [DONE]"
        yield 'data: {"type":"should-not-appear"}'

    got = [item async for item in _iter_anthropic_sse_events(lines())]
    assert got == [("", '{"type":"ping"}'), ("", "[DONE]")]
