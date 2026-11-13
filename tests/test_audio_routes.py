"""The app decides where sound comes from and where the voice goes."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from jaeger_agent.core import engine as engine_module
from jaeger_agent.core import runner
from jaeger_agent.core.config import MultimodalConfig
from jaeger_agent.core.engine import GemmaMultimodal
from jaeger_agent.core.speech import SpeechRuntime


class _Voice:
    sample_rate = 22050                     # not the 24 kHz the agent once assumed

    def synth(self, text: str):
        yield np.full(240, 0.5, dtype=np.float32)


class _Runtime:
    def run_turn(self, text: str, *, session_key: str) -> str:
        return "A short reply."

    def close(self) -> None:
        return None


class _Output:
    def __init__(self) -> None:
        self.played: list[tuple[int, int]] = []
        self.waits = self.stops = 0

    def play(self, chunk, rate) -> None:
        self.played.append((len(chunk), rate))

    def wait(self) -> None:
        self.waits += 1

    def stop(self) -> None:
        self.stops += 1


def _engine(**kw) -> GemmaMultimodal:
    engine = GemmaMultimodal(runtime=_Runtime(), **kw)
    engine.node_tts = _Voice()
    return engine


def _fake_sound_card(monkeypatch, **parts):
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(**parts))


def test_the_voice_goes_to_every_output_the_app_gives() -> None:
    speaker, lips = _Output(), _Output()
    engine = _engine(audio_mode="plain", output_mode="speech", outputs=[speaker, lips])
    engine.send_text("hello")
    assert speaker.played == lips.played == [(240, 22050)]   # the voice's own rate
    assert speaker.waits == lips.waits == 1                  # the turn ends when they do


def test_force_listen_stops_every_output() -> None:
    speaker, lips = _Output(), _Output()
    engine = _engine(audio_mode="plain", outputs=[speaker, lips])
    engine.force_listen()
    assert (speaker.stops, lips.stops) == (1, 1)


def test_with_no_outputs_given_the_voice_plays_on_the_sound_card_in_the_settings(
        monkeypatch) -> None:
    calls: list = []
    _fake_sound_card(
        monkeypatch,
        play=lambda chunk, rate, device=None: calls.append((rate, device)),
        wait=lambda: calls.append("wait"),
        stop=lambda: calls.append("stop"))
    engine = _engine(config=MultimodalConfig(
        audio_mode="plain", output_mode="speech", output_device="USB Audio"))
    engine.send_text("hello")
    assert calls == [(22050, "USB Audio"), "wait"]
    assert _engine(play=False, audio_mode="plain").outputs == []


def test_the_command_line_microphone_is_the_one_in_the_settings(monkeypatch) -> None:
    opened: dict = {}

    class Stream:
        def __init__(self, **kw) -> None:
            opened.update(kw)

        def __enter__(self):
            raise KeyboardInterrupt

        def __exit__(self, *_exc) -> bool:
            return False

    _fake_sound_card(monkeypatch, InputStream=Stream)
    engine = SimpleNamespace(FRAME=480, config=MultimodalConfig(input_device=3),
                             load=lambda: None, close=lambda: None)
    assert runner.run_mic(engine) == 0
    assert opened["device"] == 3


def test_the_echo_cancelling_device_opens_the_sound_cards_in_the_settings(
        monkeypatch) -> None:
    opened: dict = {}

    class Stream:
        def __init__(self, **kw) -> None:
            opened.update(kw)

        def start(self) -> None:
            raise RuntimeError("far enough")

    _fake_sound_card(monkeypatch, Stream=Stream)
    monkeypatch.setattr(engine_module.dxr, "pyaec", SimpleNamespace(
        Aec=lambda **_kw: SimpleNamespace(cancel_echo=lambda mic, ref: mic)))
    monkeypatch.setattr(engine_module, "SileroVad", lambda _path: object())
    engine = _engine(play=False, config=MultimodalConfig(
        audio_mode="quasi", input_device=2, output_device="Robot head"))
    part = SimpleNamespace(ready=False, tts="voice", model="hearing",
                           load=lambda *_a, **_k: None)
    engine.node_vision = engine.node_llm = engine.node_tts = engine.node_stt = part
    with pytest.raises(RuntimeError, match="far enough"):
        engine.load()
    engine.close()
    assert opened["device"] == (2, "Robot head")


def test_the_speak_tool_plays_on_the_sound_card_in_the_settings(monkeypatch) -> None:
    opened: dict = {}

    class Stream:
        def __init__(self, **kw) -> None:
            opened.update(kw)

        def start(self) -> None: ...
        def write(self, block) -> None: ...
        def stop(self) -> None: ...
        def close(self) -> None: ...

    _fake_sound_card(monkeypatch, OutputStream=Stream)
    speech = SpeechRuntime(MultimodalConfig(output_device="Robot head"))
    speech.tts, speech.stt, speech._loaded = _Voice(), object(), True
    assert speech.speak("hello") is True
    assert (opened["device"], opened["samplerate"]) == ("Robot head", 22050)
