"""Hearing starts on the phrase before the end-of-phrase wait is over."""

from __future__ import annotations

import threading

import numpy as np

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
        self.aborts: list = []
        self.called = threading.Event()

    def transcribe(self, audio, abort=None) -> str:
        self.calls.append(len(audio))
        self.aborts.append(abort)
        self.called.set()
        return f"heard pass {len(self.calls)}"


class _Runtime:
    def run_turn(self, text: str, *, session_key: str) -> str:
        return "ok"

    def close(self) -> None:
        return None


def _engine(events):
    engine = GemmaMultimodal(play=False, runtime=_Runtime(), audio_mode="plain",
                             on_event=events.append)
    engine._vad, engine.node_stt = _Vad(), _Hearing()
    return engine


def _push(engine, p: float, count: int) -> None:
    for _ in range(count):
        engine._push_frame(np.full(engine.FRAME, p, dtype=np.float32))


def test_hearing_starts_before_the_end_of_phrase_wait_is_over() -> None:
    assert 0 < EARLY < HANG             # the endpoint itself has not moved
    events: list = []
    engine = _engine(events)
    _push(engine, 1.0, VOICED)
    _push(engine, 0.0, EARLY)
    assert engine.node_stt.called.wait(2.0), "no hearing pass during the wait"
    assert not [e for e in events if e.kind == "live"], "the turn ended early"
    _push(engine, 0.0, HANG - EARLY)
    assert engine.node_stt.calls == [(VOICED + EARLY) * engine.FRAME]
    assert [e.text for e in events if e.kind == "live" and e.text] == ["heard pass 1"]


def test_an_early_pass_is_thrown_away_when_the_person_keeps_talking() -> None:
    events: list = []
    engine = _engine(events)
    _push(engine, 1.0, VOICED)
    _push(engine, 0.0, EARLY)
    assert engine.node_stt.called.wait(2.0)
    _push(engine, 1.0, 3)               # they were only taking a breath
    assert engine.node_stt.aborts[0]() is True
    _push(engine, 0.0, HANG)
    assert len(engine.node_stt.calls) == 2
    assert engine.node_stt.calls[1] == (VOICED + EARLY + 3 + EARLY) * engine.FRAME
    assert [e.text for e in events if e.kind == "live" and e.text] == ["heard pass 2"]


def test_the_live_preview_gives_way_once_the_person_goes_quiet() -> None:
    events: list = []
    engine = _engine(events)
    engine.stt = object()               # "loaded"
    _push(engine, 1.0, 20)
    engine._preview_pass()
    abort = engine.node_stt.aborts[0]
    assert abort() is False             # still talking: the preview may run
    _push(engine, 0.0, EARLY)
    assert abort() is True              # quiet: the hearing belongs to the early pass
