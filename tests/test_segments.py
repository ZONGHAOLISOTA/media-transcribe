import os
import subprocess
import sys
from pathlib import Path

import pytest

from media_transcribe.segments import build_segments, locate

SRC = Path(__file__).resolve().parents[1] / "src"


def test_module_imports_on_its_own():
    # Importing segments first must not hit a circular import through the engines.
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    proc = subprocess.run([sys.executable, "-c", "import media_transcribe.segments"],
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr


def _w(text, start, end, speaker=None):
    return {"text": text, "start": start, "end": end, "speaker": speaker}


def test_locate_partitions_the_whole_text():
    text = "Okay, so I see. Yes!"
    words = [_w("Okay", 0, 0.3), _w("so", 0.3, 0.5), _w("I", 0.6, 0.7),
             _w("see", 0.7, 1.0), _w("Yes", 1.5, 1.8)]
    spans = locate(text, words)
    assert "".join(text[s:e] for s, e in spans) == text
    assert text[slice(*spans[0])] == "Okay,"


def test_sentences_get_punctuation_and_times():
    text = "你好。今天讲数据。"
    words = [_w(c, i * 0.2, i * 0.2 + 0.2) for i, c in enumerate("你好今天讲数据")]
    segs = build_segments(text, words)
    assert [s.text for s in segs] == ["你好。", "今天讲数据。"]
    assert (segs[1].start, segs[1].end) == pytest.approx((0.4, 1.4))


def test_unmatched_word_loses_no_text():
    text = "We use C++ and Python."
    words = [_w("We", 0, 0.2), _w("use", 0.2, 0.4), _w("Cplusplus", 0.4, 0.8),
             _w("and", 0.8, 1.0), _w("Python", 1.0, 1.4)]
    segs = build_segments(text, words)
    assert "".join(s.text for s in segs).replace(" ", "") == text.replace(" ", "")


def test_unmatched_tail_word_keeps_its_text():
    words = [_w(c, i * 0.2, i * 0.2 + 0.2) for i, c in enumerate("我们用的是")]
    segs = build_segments("我们用的是GPT-4", words + [_w("XYZ", 1.0, 1.3)])
    assert [s.text for s in segs] == ["我们用的是GPT-4"]


def test_no_word_matching_keeps_all_text():
    assert [s.text for s in build_segments("Hello world", [_w("x", 0, .2), _w("y", .2, .4)])] \
        == ["Hello world"]


@pytest.mark.parametrize("text,token", [
    ("The price is 3.5. Okay, thanks.", "35"),
    ("The model is GPT-4. Okay, thanks.", "GPT4"),
    ("We meet at 9:30. Okay, thanks.", "930"),
])
def test_aligner_normalized_tokens_still_match(text, token):
    head = text.split(". ")[0].split()[:-1]
    words = [_w(t, i * 0.3, i * 0.3 + 0.25, "A") for i, t in enumerate(head + [token])]
    words += [_w("Okay", 3.0, 3.3, "B"), _w("thanks", 3.3, 3.8, "B")]
    segs = build_segments(text, words)
    assert [(s.speaker, s.text) for s in segs] == [("A", text.split(" Okay")[0]),
                                                   ("B", "Okay, thanks.")]


def test_unmatched_word_keeps_its_sentence_end_and_speaker():
    text = "我们这次用的是ZZ-9。好的。"
    words = [_w(c, i * 0.2, i * 0.2 + 0.2, "A") for i, c in enumerate("我们这次用的是")]
    words += [_w("QQ", 1.4, 1.8, "A"), _w("好", 2.5, 2.7, "B"), _w("的", 2.7, 2.9, "B")]
    segs = build_segments(text, words)
    assert [(s.speaker, s.text) for s in segs] == [("A", "我们这次用的是ZZ-9。"), ("B", "好的。")]


def test_short_latin_word_does_not_latch_onto_a_longer_word():
    text = "the cat sat"
    spans = locate(text, [_w("the", 0, .1), _w("a", .1, .2), _w("cat", .2, .4), _w("sat", .4, .6)])
    assert text[slice(*spans[2])].strip() == "cat"


def test_decimal_point_is_not_a_sentence_end():
    text = "It costs 3.5 dollars."
    words = [_w("It", 0, 0.2), _w("costs", 0.2, 0.5), _w("3", 0.5, 0.6),
             _w("5", 0.6, 0.7), _w("dollars", 0.7, 1.0)]
    assert [s.text for s in build_segments(text, words)] == [text]


def test_period_glued_to_a_capital_still_ends_the_sentence():
    text = "Open the file.It works now."
    words = [_w(t, i * 0.3, i * 0.3 + 0.25) for i, t in
             enumerate(["Open", "the", "file", "It", "works", "now"])]
    assert [s.text for s in build_segments(text, words)] == ["Open the file.", "It works now."]


def test_speaker_change_splits_segments():
    text = "How are you? Fine, thanks."
    words = [_w("How", 0, 0.3, "A"), _w("are", 0.3, 0.5, "A"), _w("you", 0.5, 0.8, "A"),
             _w("Fine", 1.0, 1.4, "B"), _w("thanks", 1.4, 2.0, "B")]
    segs = build_segments(text, words)
    assert [(s.speaker, s.text) for s in segs] == [("A", "How are you?"), ("B", "Fine, thanks.")]


def test_flickering_labels_inside_a_sentence_are_smoothed():
    # Labels flicker on single characters; the sentence belongs to speaker 1.
    text = "然后我们再看一下这个方案的效果。"
    labels = "111110111010101"
    words = [_w(c, i * 0.15, i * 0.15 + 0.12, lab)
             for i, (c, lab) in enumerate(zip("然后我们再看一下这个方案的效果", labels))]
    segs = build_segments(text, words)
    assert [(s.speaker, s.text) for s in segs] == [("1", text)]


def test_clipped_sentence_edge_rejoins_its_speaker():
    text = "这个问题不大的。然后呢？"
    words = [_w(c, i * 0.2, i * 0.2 + 0.2, "A") for i, c in enumerate("这个问题不大")]
    words += [_w("的", 1.2, 1.3, "B"), _w("然", 1.5, 1.7, "B"), _w("后", 1.7, 1.9, "B"),
              _w("呢", 1.9, 2.3, "B")]
    segs = build_segments(text, words)
    assert [(s.speaker, s.text) for s in segs] == [("A", "这个问题不大的。"), ("B", "然后呢？")]


def test_short_whole_sentence_keeps_its_own_speaker():
    text = "Open the file now. Yeah. Then run the tests."
    words = [_w("Open", 0, 0.3, "A"), _w("the", 0.3, 0.4, "A"), _w("file", 0.4, 0.5, "A"),
             _w("now", 0.5, 0.9, "A"), _w("Yeah", 1.0, 1.2, "B"),
             _w("Then", 1.4, 1.6, "A"), _w("run", 1.6, 1.8, "A"), _w("the", 1.8, 1.9, "A"),
             _w("tests", 1.9, 2.1, "A")]
    segs = build_segments(text, words)
    assert [s.speaker for s in segs] == ["A", "B", "A"]


def test_no_words_gives_one_untimed_segment():
    segs = build_segments("只有文字。", [])
    assert len(segs) == 1 and segs[0].text == "只有文字。"
