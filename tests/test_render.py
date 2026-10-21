import json

import pytest

from media_transcribe.engines.base import Options, Segment
from media_transcribe.render import needs_segments, render_segments, render_text

DIALOG = [
    Segment("我们开始吧。", 0.0, 1.5, "SPEAKER_01"),
    Segment("好的。", 1.6, 2.0, "SPEAKER_00"),
    Segment("先看数据。", 2.1, 3.0, "SPEAKER_01"),
    Segment("这里用 API。", 3.1, 65.2, "SPEAKER_01"),
]


def test_speakers_numbered_by_first_appearance_in_chinese():
    out = render_segments(DIALOG, fmt="txt", language="zh", timestamps=False)
    assert out == ("说话人 1：我们开始吧。\n\n说话人 2：好的。\n\n"
                   "说话人 1：先看数据。这里用 API。\n")


@pytest.mark.parametrize("code", ["zh-Hant", "zh_CN", "zh-HK", "Chinese", "yue"])
def test_chinese_variants_get_chinese_labels(code):
    out = render_segments(DIALOG[:1], fmt="txt", language=code, timestamps=False)
    assert out.startswith("说话人 1：")


def test_english_labels_and_timestamps():
    segs = [Segment("Hi there.", 61.0, 62.0, "A"), Segment("Hello.", 62.5, 63.0, "B")]
    out = render_segments(segs, fmt="txt", language="en", timestamps=True)
    assert out == "[00:01:01] Speaker 1: Hi there.\n\n[00:01:02] Speaker 2: Hello.\n"


def test_english_turn_joins_sentences_with_a_space():
    segs = [Segment("Save it.", 0, 1, "A"), Segment("Then send it.", 1, 2, "A")]
    out = render_segments(segs, fmt="txt", language="en", timestamps=False)
    assert out == "Speaker 1: Save it. Then send it.\n"


def test_timestamps_only_is_one_line_per_sentence():
    segs = [Segment("第一句。", 0, 1), Segment("第二句。", 3725.4, 3727)]
    out = render_segments(segs, fmt="txt", language="zh", timestamps=True)
    assert out == "[00:00:00] 第一句。\n[01:02:05] 第二句。\n"


def test_srt_and_vtt():
    srt = render_segments(DIALOG[:2], fmt="srt", language="zh", timestamps=False)
    assert srt.startswith("1\n00:00:00,000 --> 00:00:01,500\n说话人 1：我们开始吧。\n\n2\n")
    vtt = render_segments(DIALOG[:1], fmt="vtt", language="zh", timestamps=False)
    assert vtt == "WEBVTT\n\n00:00:00.000 --> 00:00:01.500\n说话人 1：我们开始吧。\n"


def test_zero_length_cue_is_stretched_but_stops_at_the_next():
    segs = [Segment("Use it.", 0.0, 1.0, "A"), Segment("Yeah.", 1.2, 1.2, "B"),
            Segment("Then.", 1.6, 2.0, "A")]
    srt = render_segments(segs, fmt="srt", language="en", timestamps=False)
    assert "00:00:01,200 --> 00:00:01,600\nSpeaker 2: Yeah." in srt
    assert "00:00:01,600 --> 00:00:02,600\nSpeaker 1: Then." in srt


def test_txt_without_speakers_or_timestamps_is_running_text():
    segs = [Segment("One.", 0, 1), Segment("Two.", 1, 2)]
    assert render_segments(segs, fmt="txt", language="en", timestamps=False) == "One. Two.\n"


def test_tsv_and_json():
    tsv = render_segments(DIALOG[:1], fmt="tsv", language="zh", timestamps=False)
    assert tsv.splitlines() == ["start\tend\tspeaker\ttext", "0.000\t1.500\t说话人 1\t我们开始吧。"]
    data = json.loads(render_segments(DIALOG, fmt="json", language="zh", timestamps=False))
    assert data["text"] == "我们开始吧。好的。先看数据。这里用 API。"
    assert data["segments"][1] == {"start": 1.6, "end": 2.0, "speaker": "说话人 2", "text": "好的。"}


def test_plain_text_formats():
    assert render_text("全文。\n", fmt="txt", language="zh") == "全文。\n"
    assert json.loads(render_text("全文。\n", fmt="json", language="zh")) == {
        "language": "zh", "text": "全文。", "segments": []}


@pytest.mark.parametrize("opts,fmt,expected", [
    (Options(), "txt", False),
    (Options(), "json", False),
    (Options(), "srt", True),
    (Options(timestamps=True), "txt", True),
    (Options(diarize=True), "txt", True),
])
def test_needs_segments(opts, fmt, expected):
    assert needs_segments(opts, fmt) is expected
