from media_transcribe.clean import clean, clean_segments
from media_transcribe.engines.base import Segment


def test_collapses_consecutive_duplicates():
    assert clean("你好。你好。你好。今天做菜。") == "你好。今天做菜。\n"


def test_strips_trailing_context_echo():
    ctx = ["炒菜", "刀工", "切块", "调料"]
    out = clean("正文内容。炒菜 刀工 切块 调料。", ctx)
    assert "刀工" not in out
    assert out.startswith("正文内容。")


def test_no_context_means_no_tail_strip():
    out = clean("正文内容。炒菜 刀工 切块。", None)
    assert "炒菜 刀工 切块" in out


def test_empty_input():
    assert clean("") == ""


def test_trailing_newline_added():
    assert clean("一句话。").endswith("\n")


def test_strips_repeated_trailing_filler():
    assert clean("正文。嗯。嗯。嗯。") == "正文。\n"


def test_strips_mixed_trailing_filler():
    assert clean("正文。嗯。呃。") == "正文。\n"


def test_keeps_real_content_with_filler_char_inside():
    # A real sentence that merely contains a filler char is not dropped.
    out = clean("我们开始吧。")
    assert "我们开始吧" in out


def test_segments_merge_consecutive_duplicates_of_one_speaker():
    segs = [Segment("你好。", 0, 1, "A"), Segment("你好。", 1, 2, "A"), Segment("你好。", 2, 3, "B")]
    out = clean_segments(segs)
    assert [(s.speaker, s.text, s.end) for s in out] == [("A", "你好。", 2), ("B", "你好。", 3)]


def test_segments_strip_junk_tail_and_empty():
    ctx = ["模型", "数据", "训练", "接口"]
    segs = [Segment("正文。", 0, 1, "A"), Segment(" ", 1, 2, "A"),
            Segment("模型 数据 训练。", 2, 3, "A"), Segment("嗯。", 3, 4, "B")]
    assert [s.text for s in clean_segments(segs, ctx)] == ["正文。"]
