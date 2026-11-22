"""An open mic (half_duplex=False): the engine keeps hearing while it answers,
a person can cut in, and every turn says when it is over."""

from __future__ import annotations

import threading

import numpy as np

from jaeger_agent.core.config import MultimodalConfig
from jaeger_agent.core.engine import GemmaMultimodal
from jaeger_agent.core.policy import (
    EARLY_DECODE_MS, FRAME_MS, MIN_SPEECH_MS, PRE_ROLL_MS, SILENCE_HANGOVER_MS,
)

EARLY = EARLY_DECODE_MS // FRAME_MS
HANG = SILENCE_HANGOVER_MS // FRAME_MS
VOICED = MIN_SPEECH_MS // FRAME_MS + 2
PRE = PRE_ROLL_MS // FRAME_MS


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
    engine._sounded = 1.0               # she has begun to speak (before any sound it would be a pause: see below)
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
    engine._sounded = 1.0               # she is speaking
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


# -- a pause is not the end of what they were saying -------------------------
#
# The end of a phrase is half a second of quiet. A person thinking of their
# next words is quiet for longer, and what they have said so far often reads
# as a whole sentence. So until she has made a sound of her answer, a person
# who goes on talking was not finished: the answer is dropped, the sound of
# what they said is kept, and it is heard as one with what they say next.

ONE = VOICED + HANG                     # a whole phrase, in frames, with the quiet that ended it


class _Lengths(_Hearing):
    """Hears how much sound it was handed: "<frames> frames"."""

    def transcribe(self, audio, abort=None) -> str:
        self.calls.append(len(audio))
        return f"{len(audio) // (16000 * FRAME_MS // 1000)} frames"


def _frames(text: str) -> int:
    return int(text.split()[0])


def test_going_on_before_she_has_made_a_sound_is_heard_as_one_with_what_was_said() -> None:
    events, runtime = _Events(), _SlowRuntime()
    engine = _engine(events, runtime)
    engine.node_stt = _Lengths()
    _phrase(engine)                     # "Let me tell you a long story"
    assert runtime.entered.wait(5)
    _push(engine, 1.0, VOICED)          # "...about the time I got lost": they go on
    assert runtime.cut.wait(2), "the answer to the first half went on being written"
    _push(engine, 0.0, HANG)
    events.wait(2)
    first, whole = runtime.asked
    assert _frames(first) == VOICED + EARLY
    assert _frames(whole) >= ONE + VOICED + EARLY                 # both parts, in one pass of hearing
    assert events.told() == [
        ("user", first), ("turn", "end"),
        ("user", whole), ("assistant", f"Reply to {whole}."), ("turn", "end")]


def test_a_closed_mic_that_listens_while_she_thinks_hears_it_as_one_too() -> None:
    events, runtime = _Events(), _SlowRuntime()
    engine = GemmaMultimodal(
        play=False, runtime=runtime, audio_mode="plain", on_event=events.append,
        half_duplex=True, listens_while_thinking=True, output_mode="speech",
        config=MultimodalConfig(audio_mode="plain", output_mode="speech", wake_required=False))
    engine._vad, engine.node_stt, engine.node_tts = _Vad(), _Lengths(), _Voice()
    _phrase(engine)                     # comes straight back: she thinks on a thread of her own
    assert runtime.entered.wait(5)
    _phrase(engine)
    events.wait(2)
    assert len(runtime.asked) == 2 and _frames(runtime.asked[1]) >= ONE + VOICED + EARLY


def test_a_noise_while_she_thinks_does_not_lose_what_was_said() -> None:
    events, runtime = _Events(), _SlowRuntime()
    engine = _engine(events, runtime)
    _phrase(engine)
    assert runtime.entered.wait(5)
    engine.node_stt.transcribe = lambda audio, abort=None: "phrase 1"   # a cough after it: the words in the sound are still theirs
    _phrase(engine)
    events.wait(2)
    assert runtime.asked == ["phrase 1", "phrase 1"]              # asked again: the first answer was dropped for the noise
    assert events.told()[-3:] == [
        ("user", "phrase 1"), ("assistant", "Reply to phrase 1."), ("turn", "end")]


def test_a_click_too_short_to_be_speech_does_not_stop_her_answer() -> None:
    events, runtime = _Events(), _SlowRuntime()
    engine = _engine(events, runtime)
    _phrase(engine)
    assert runtime.entered.wait(5)
    _push(engine, 1.0, 2)               # 60 ms of sound
    assert not runtime.cut.wait(0.2)
    runtime.release.set()
    events.wait(1)
    assert events.told() == [("user", "phrase 1"), ("assistant", "Reply to phrase 1."), ("turn", "end")]


def test_going_on_while_the_first_words_are_still_being_made_out_is_heard_as_one_too() -> None:
    """Hearing takes a moment. A person who goes on before their first words
    have even been made out was pausing just the same."""
    events, runtime = _Events(), _SlowRuntime()
    runtime.release.set()
    engine = _engine(events, runtime)
    made_out = threading.Event()

    unwanted = []

    class SlowEars(_Lengths):
        def transcribe(self, audio, abort=None) -> str:
            if not self.calls:
                made_out.wait(3)        # the first phrase takes a while to make out
                unwanted.append(abort())
            return super().transcribe(audio, abort)

    engine.node_stt = SlowEars()
    _push(engine, 1.0, VOICED)
    _push(engine, 0.0, HANG)            # the first phrase ends; hearing starts on it
    _push(engine, 1.0, VOICED)          # they go on before it is made out
    made_out.set()
    _push(engine, 0.0, HANG)
    events.wait(1)
    assert unwanted == [True]                                     # the pass on the first part alone is told nobody needs it
    (whole,) = runtime.asked                                      # one answer, to all of it
    assert _frames(whole) >= ONE + VOICED + EARLY
    assert events.told() == [("user", whole), ("assistant", f"Reply to {whole}."), ("turn", "end")]


def test_a_sound_too_short_to_be_a_phrase_is_let_go_after_the_quiet() -> None:
    """Left open, it would take in all the quiet up to the next thing said,
    and hearing makes words up out of a long quiet ("Thank you.")."""
    events, runtime = _Events(), _SlowRuntime()
    runtime.release.set()
    engine = _engine(events, runtime)
    _push(engine, 1.0, 2)               # a click
    _push(engine, 0.0, HANG)
    assert not engine._in_speech
    _push(engine, 0.0, 300)             # nine seconds of quiet
    _phrase(engine)
    events.wait(1)
    assert engine.node_stt.calls == [(PRE + VOICED + EARLY) * engine.FRAME]   # heard on its own, with a breath before it, not the quiet


def test_the_start_of_the_first_word_comes_with_the_phrase() -> None:
    """The detector opens a moment into the first word. Without the sound just
    before it, a phrase lost its first word or two ("I saw a dog" was heard as
    "a dog", measured 2026-11-22)."""
    events, runtime = _Events(), _SlowRuntime()
    runtime.release.set()
    engine = _engine(events, runtime)
    heard = []
    engine.node_stt.transcribe = lambda audio, abort=None: heard.append(audio.copy()) or "words"
    _push(engine, 0.0, 50)              # quiet
    _push(engine, 0.2, 3)               # the start of a word, under the bar that opens the phrase
    _push(engine, 1.0, VOICED)
    _push(engine, 0.0, HANG)
    events.wait(1)
    frames = heard[-1].reshape(-1, engine.FRAME)[:, 0]
    assert len(frames) == PRE + VOICED + EARLY                     # heard from the early pass; not the whole quiet before it
    assert list(frames[PRE - 3:PRE]) == [0.2] * 3                  # the start of the word is there
