"""An open mic (half_duplex=False): the engine keeps hearing while it answers,
a person can cut in, and every turn says when it is over."""

from __future__ import annotations

import threading

import numpy as np

from jaeger_agent.core.config import MultimodalConfig
from jaeger_agent.core.engine import GemmaMultimodal
from jaeger_agent.core.policy import (
    EARLY_DECODE_MS, FRAME_MS, MIN_SPEECH_MS, SILENCE_HANGOVER_MS,
)

EARLY = EARLY_DECODE_MS // FRAME_MS
HANG = SILENCE_HANGOVER_MS // FRAME_MS
VOICED = MIN_SPEECH_MS // FRAME_MS + 2


class _Vad:
    def probability(self, frame) -> float:
        return float(frame[0])          # the test writes the answer into the frame


class _Hearing:
    def __init__(self) -> None:
        self.calls: list[int] = []      # samples handed to each pass

    def transcribe(self, audio, abort=None) -> str:
        self.calls.append(len(audio))
        return f"phrase {len(self.calls)}"


class _Voice:
    def synth(self, text: str):
        yield np.zeros(240, dtype=np.float32)

    def set_voice(self, voice: str) -> None:
        return None


class _SlowRuntime:
    """Its first reply takes until it is let go (``release``) or cut (``interrupt``)."""

    def __init__(self) -> None:
        self.asked: list[str] = []
        self.entered, self.release, self.cut = (threading.Event() for _ in range(3))

    def run_turn(self, text: str, *, session_key: str) -> str:
        self.asked.append(text)
        if len(self.asked) == 1:
            self.entered.set()
            for _ in range(300):        # 3 s at most, so a mic that is not open fails the test, not hangs it
                if self.release.is_set() or self.cut.is_set():
                    break
                self.release.wait(0.01)
        return f"Reply to {text}."

    def interrupt(self, *, session_key: str) -> None:
        self.cut.set()

    def close(self) -> None:
        return None


class _Events(list):
    """What the engine did, and a way to wait for it."""

    def __init__(self) -> None:
        super().__init__()
        self.came = threading.Condition()

    def append(self, event) -> None:
        with self.came:
            super().append(event)
            self.came.notify_all()

    def wait(self, count: int, kind: str = "turn") -> None:
        with self.came:
            assert self.came.wait_for(
                lambda: sum(e.kind == kind for e in self) >= count, 5), \
                f"waited for {count} × {kind}, got {self.told()}"

    def told(self) -> list:
        return [(e.kind, e.text) for e in self if e.kind in ("user", "assistant", "turn")]


def _engine(events, runtime, *, half_duplex: bool = False):
    engine = GemmaMultimodal(
        play=False, runtime=runtime, audio_mode="plain", on_event=events.append,
        half_duplex=half_duplex, output_mode="speech",
        config=MultimodalConfig(audio_mode="plain", output_mode="speech", wake_required=False))
    engine._vad, engine.node_stt, engine.node_tts = _Vad(), _Hearing(), _Voice()
    return engine


def _push(engine, p: float, count: int) -> None:
    for _ in range(count):
        engine._push_frame(np.full(engine.FRAME, p, dtype=np.float32))


def _phrase(engine) -> None:
    _push(engine, 1.0, VOICED)
    _push(engine, 0.0, HANG)


def test_a_person_can_cut_in_while_the_reply_is_being_written() -> None:
    events, runtime = _Events(), _SlowRuntime()
    engine = _engine(events, runtime)
    _phrase(engine)                     # comes straight back: the mic stays open
    assert runtime.entered.wait(5), "the first phrase was not answered"
    _phrase(engine)                     # said over the reply
    events.wait(2)
    assert runtime.cut.is_set(), "the reply on its way was not cut"
    # the cut reply is over before the words that cut it are shown
    assert events.told() == [
        ("user", "phrase 1"), ("turn", "end"),
        ("user", "phrase 2"), ("assistant", "Reply to phrase 2."), ("turn", "end")]


def test_a_reply_ending_does_not_wipe_what_an_open_mic_is_hearing() -> None:
    events, runtime = _Events(), _SlowRuntime()
    engine = _engine(events, runtime)
    _phrase(engine)
    assert runtime.entered.wait(5)
    _push(engine, 1.0, VOICED)          # someone starts talking as the reply ends
    runtime.release.set()
    events.wait(1)
    _push(engine, 0.0, HANG)
    events.wait(2)
    assert engine.node_stt.calls[1] == (VOICED + EARLY) * engine.FRAME
    assert events.told()[-3:] == [
        ("user", "phrase 2"), ("assistant", "Reply to phrase 2."), ("turn", "end")]


def test_the_sound_stopping_ends_a_phrase() -> None:
    events, runtime = _Events(), _SlowRuntime()
    runtime.release.set()
    engine = _engine(events, runtime)
    engine.end_phrase()                 # nothing forming: nothing happens
    _push(engine, 1.0, VOICED)          # then no quiet ever comes: the mic was gated or closed
    assert events.told() == []
    engine.end_phrase()
    events.wait(1)
    assert events.told() == [
        ("user", "phrase 1"), ("assistant", "Reply to phrase 1."), ("turn", "end")]
    assert engine.node_stt.calls == [VOICED * engine.FRAME]


def test_a_typed_turn_and_a_closed_mic_turn_say_when_they_are_over() -> None:
    events, runtime = _Events(), _SlowRuntime()
    runtime.release.set()
    engine = _engine(events, runtime, half_duplex=True)
    engine.send_text("hello")
    _phrase(engine)                     # half duplex: answered before the call comes back
    assert events.told() == [
        ("user", "hello"), ("assistant", "Reply to hello."), ("turn", "end"),
        ("user", "phrase 1"), ("assistant", "Reply to phrase 1."), ("turn", "end")]


def test_a_new_voice_or_a_line_of_sound_waits_for_the_turn_to_end() -> None:
    events, runtime = _Events(), _SlowRuntime()
    engine = _engine(events, runtime)
    _phrase(engine)
    assert runtime.entered.wait(5)
    done: list[str] = []
    others = [threading.Thread(target=lambda: (list(engine.synth("a line")), done.append("line"))),
              threading.Thread(target=lambda: (engine._floor.acquire(), engine._floor.release(), done.append("floor")))]
    for t in others:
        t.start()
    others[0].join(0.2)
    assert done == [], "a line was voiced beside the reply"
    runtime.release.set()
    for t in others:
        t.join(5)
    assert sorted(done) == ["floor", "line"]


def test_a_sound_with_no_words_in_it_gets_a_verdict() -> None:
    """Whoever plays the sound may be holding the agent's line while a person
    seems to talk over it: it must be told when that was nobody."""
    events, runtime = _Events(), _SlowRuntime()
    engine = _engine(events, runtime, half_duplex=True)
    engine.node_stt.transcribe = lambda audio, abort=None: ""     # a cough
    _phrase(engine)
    assert [e.text for e in events if e.kind == "state"][-1] == "no words — ignored"
    assert events.told() == []
