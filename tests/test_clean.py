import pytest

from media_transcribe.clean import clean, clean_segments
from media_transcribe.engines.base import Segment

CTX = ["Kubernetes", "Docker", "Helm"]


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


def test_english_transcript_mentioning_context_terms_is_kept():
    # Regression: the whole English text is one "sentence" here and used to be wiped.
    ctx = ["Kubernetes", "Docker", "Helm"]
    text = "Today we compare Kubernetes and Docker and Helm for small teams. That is the plan."
    assert clean(text, ctx).strip() == text


def test_real_last_sentence_naming_terms_is_kept():
    ctx = ["Kubernetes", "Docker", "Helm"]
    out = clean("正文。今天对比 Kubernetes 和 Docker 还有 Helm 三个工具。", ctx)
    assert "三个工具" in out


def test_english_context_echo_is_still_stripped():
    out = clean("正文。Kubernetes, Docker, Helm.", ["Kubernetes", "Docker", "Helm"])
    assert out == "正文。\n"


@pytest.mark.parametrize("text", [
    "Kubernetes, Docker, and Helm.",          # a real sentence has some other word
    "We use Kubernetes, Docker and Helm.",
    "Helm, Docker, Kubernetes.",              # not in the context's order: a real list
])
def test_real_speech_naming_terms_is_never_wiped(text):
    assert clean(text, CTX).strip() == text


def test_real_final_answer_survives_in_segments():
    segs = [Segment("What do you run in production?", 0, 2, "A"),
            Segment("Kubernetes, Docker, and Helm.", 2.5, 4, "B")]
    assert [s.speaker for s in clean_segments(segs, CTX)] == ["A", "B"]


def test_punctuated_context_list_echo_is_stripped():
    ctx = ["Kubernetes,", "Docker,", "Helm,", "Terraform"]
    assert clean("谢谢大家。Kubernetes, Docker, Helm, Terraform。", ctx) == "谢谢大家。\n"


def test_symbol_terms_echo_is_stripped():
    assert clean("谢谢。C++ C# .NET Java。", ["C++", "C#", ".NET", "Java"]) == "谢谢。\n"


def test_repeated_echo_wrapping_around_the_context_is_stripped():
    assert clean("正文。Kubernetes Docker Helm Kubernetes Docker。", CTX) == "正文。\n"


def test_echo_split_across_segments_is_stripped_whole():
    ctx = ["Kubernetes", "Docker", "Helm", "Ansible", "Prometheus", "Grafana"]
    segs = [Segment("正文。", 0, 1, "A"),
            Segment("Kubernetes, Docker, Helm, Ansible,", 1, 7, "A"),
            Segment("Prometheus, Grafana.", 7, 9, "A")]
    assert [s.text for s in clean_segments(segs, ctx)] == ["正文。"]


def test_segments_keep_english_repeats_like_plain_clean():
    segs = [Segment("Okay.", 0, 1, "A"), Segment("Okay.", 1, 2, "A")]
    assert [s.text for s in clean_segments(segs)] == ["Okay.", "Okay."]


def test_segments_merge_consecutive_duplicates_of_one_speaker():
    segs = [Segment("你好。", 0, 1, "A"), Segment("你好。", 1, 2, "A"), Segment("你好。", 2, 3, "B")]
    out = clean_segments(segs)
    assert [(s.speaker, s.text, s.end) for s in out] == [("A", "你好。", 2), ("B", "你好。", 3)]


def test_segments_strip_junk_tail_and_empty():
    ctx = ["模型", "数据", "训练", "接口"]
    segs = [Segment("正文。", 0, 1, "A"), Segment(" ", 1, 2, "A"),
            Segment("模型 数据 训练。", 2, 3, "A"), Segment("嗯。", 3, 4, "B")]
    assert [s.text for s in clean_segments(segs, ctx)] == ["正文。"]
