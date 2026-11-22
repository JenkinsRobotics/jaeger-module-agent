"""HOLDING (structured audio mode): a phrase that does not read as a finished
thought is not answered yet. What the person says next joins it, and the two
are answered as one. This is what keeps a pause in the middle of a sentence
from being taken for its end."""

from __future__ import annotations

import threading
import time

import numpy as np

from jaeger_agent.core.config import MultimodalConfig
from jaeger_agent.core.engine import GemmaMultimodal
from jaeger_agent.core.policy import SAMPLE_RATE


class _Runtime:
    """Answers at once, or (``slow``) only when let go."""

    def __init__(self, slow: bool = False) -> None:
        self.turns: list[str] = []
        self.entered, self.release = threading.Event(), threading.Event()
        if not slow:
            self.release.set()

    def run_turn(self, text: str, *, session_key: str) -> str:
        self.turns.append(text)
        self.entered.set()
        self.release.wait(3)              # a turn that is never let go fails the test, not hangs it
        return "answer"

    def note(self, session_key: str, user_text: str, reply: str) -> None:
        return None

    def close(self) -> None:
        return None


class _Voice:
    def synth(self, text: str):
        yield np.zeros(240, dtype=np.float32)

    def set_voice(self, voice: str) -> None:
        return None


class _Ears:
    words = ""

    def transcribe(self, audio, abort=None) -> str:
        return self.words


def _engine(runtime, events, *, half_duplex: bool = True):
    engine = GemmaMultimodal(
        play=False, runtime=runtime, audio_mode="structured", output_mode="speech",
        half_duplex=half_duplex, on_event=events.append,
        config=MultimodalConfig(wake_required=False))
    engine.node_tts, engine.node_stt = _Voice(), _Ears()
    return engine


def _hears(engine, words: str) -> None:
    engine.node_stt.words, engine._started = words, time.time()
    # three seconds of sound: a pace a person speaks at (a few words in a
    # tenth of a second is dropped as echo garble)
    engine._on_turn(np.zeros(3 * SAMPLE_RATE, dtype=np.float32))


def test_a_thought_that_is_not_finished_is_held_and_answered_with_its_end() -> None:
    runtime, events = _Runtime(), []
    engine = _engine(runtime, events)

    _hears(engine, "I went to the market and")           # a pause in the middle of the sentence
    assert runtime.turns == []
    assert [e.text for e in events if e.kind == "state"][-1].startswith("HOLDING")

    _hears(engine, "bought some fresh bread.")
    assert len(runtime.turns) == 1                       # one answer, to the whole sentence
    assert runtime.turns[0].endswith("I went to the market and bought some fresh bread.")
    assert [e.text for e in events if e.kind == "user"] == [
        "I went to the market and bought some fresh bread."]   # and shown whole, not as its second half


def test_a_thought_that_reads_as_finished_is_answered_at_once() -> None:
    runtime, events = _Runtime(), []
    engine = _engine(runtime, events)
    _hears(engine, "I went to the market.")
    assert len(runtime.turns) == 1


def test_a_scene_that_waits_takes_their_line_as_it_was_said() -> None:
    runtime, events = _Runtime(), []
    engine = _engine(runtime, events)
    engine.perform("say Hi, who are you? | listen My name is Sam | say Nice to meet you!")
    engine._scene_from = 0.0                             # her line has been played
    events.clear()

    _hears(engine, "My name is")                         # not a finished thought, but it is their line
    assert [e.text for e in events if e.kind == "scene"][:1] == ["heard"]
    assert [e.text for e in events if e.kind == "audio"] == ["Nice to meet you!"]
    assert runtime.turns == [] and engine._hold["text"] == ""


def test_an_open_mic_goes_on_hearing_while_a_held_thought_is_answered() -> None:
    runtime, events = _Runtime(slow=True), []
    engine = _engine(runtime, events, half_duplex=False)
    _hears(engine, "I went to the market and")
    engine._hold["deadline"] = 0.0                       # nothing more came: the hold has run out

    engine._tick_hold()                                  # called by whoever brings the sound
    assert runtime.entered.wait(2)                       # the reply is being written,
    assert not [e for e in events if e.kind == "turn"]   # and the caller was not kept until its end
    runtime.release.set()
    for _ in range(200):
        if [e for e in events if e.kind == "turn"]:
            break
        time.sleep(0.01)
    assert [e.text for e in events if e.kind == "turn"] == ["end"]
    assert runtime.turns[0].endswith("I went to the market and")
