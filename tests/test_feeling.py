"""The voice is told how each line feels.

The model already picks a feeling for the face with each reply (a body's
``avatar`` tool). The same feeling goes to the voice with every line of that
reply, so a voice that can act on it sounds as the face looks. A voice that
cannot is asked the old way and reads the line as before.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from jaeger_agent.core import engine as engine_module
from jaeger_agent.core.config import MultimodalConfig
from jaeger_agent.core.engine import GemmaMultimodal


class _ActingVoice:
    """A voice that can act: its synth takes a feeling."""

    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []

    def synth(self, text: str, *, feeling: str = ""):
        self.lines.append((text, feeling))
        yield np.zeros(240, dtype=np.float32)


class _PlainVoice:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def synth(self, text: str):
        self.lines.append(text)
        yield np.zeros(240, dtype=np.float32)


class _Runtime:
    """Answers; ``feels`` is what its avatar tool call would tell the engine."""

    def __init__(self) -> None:
        self.engine, self.feels = None, ""

    def run_turn(self, text: str, *, session_key: str) -> str:
        if self.feels:
            self.engine.feel(self.feels)      # as a body's avatar tool does when the body takes the emotion
        return "Oh no. That is sad."

    def note(self, session_key: str, user_text: str, reply: str) -> None:
        return None

    def close(self) -> None:
        return None


def _engine(voice, events=None):
    runtime = _Runtime()
    engine = GemmaMultimodal(
        play=False, runtime=runtime, audio_mode="plain", output_mode="speech",
        on_event=(events.append if events is not None else None),
        config=MultimodalConfig(wake_required=False))
    engine.node_tts = voice
    runtime.engine = engine
    return engine, runtime


def test_the_feeling_of_a_reply_goes_to_the_voice_with_its_lines() -> None:
    voice, events = _ActingVoice(), []
    engine, runtime = _engine(voice, events)
    runtime.feels = "sad"
    engine.send_text("My fish died.")
    assert voice.lines and {feeling for _, feeling in voice.lines} == {"sad"}
    assert ("feeling", "sad") in [(e.kind, e.text) for e in events]      # anyone watching can see what the voice was told

    voice.lines.clear()
    runtime.feels = ""                         # the next reply picks no feeling
    engine.send_text("Anyway.")
    assert {feeling for _, feeling in voice.lines} == {""}                # it is not carried over from the last one


def test_a_voice_that_cannot_act_is_asked_the_old_way() -> None:
    voice = _PlainVoice()
    engine, runtime = _engine(voice)
    runtime.feels = "sad"
    engine.send_text("My fish died.")
    assert voice.lines                         # spoken, with no feeling it has no place for


def test_a_scenes_feeling_goes_to_the_lines_after_it() -> None:
    voice = _ActingVoice()
    engine, _ = _engine(voice)
    engine.perform("say Hello there. | emotion sad | say Oh no. | say It is gone. | say It came back! | emotion joy | on back")
    assert voice.lines == [("Hello there.", ""), ("Oh no.", "sad"), ("It is gone.", "sad"),
                           ("It came back!", "sad")]                     # a move put on a word is not the whole line's feeling


def test_the_voice_node_hands_the_feeling_to_a_module_that_takes_it(monkeypatch) -> None:
    import jaeger_os.core.modules as modules

    asked = []

    class Acting:
        compute = "cpu"

        def synth(self, text, *, voice=None, feeling=""):
            asked.append((text, feeling))
            yield np.asarray([0.5], dtype=np.float32)

    class Plain:
        compute = "cpu"

        def synth(self, text):
            asked.append((text,))
            yield np.asarray([0.5], dtype=np.float32)

    for module, expect in ((Acting, [("Oh,", "sad"), ("Oh,", "joy")]), (Plain, [("Oh,",)])):
        asked.clear()
        monkeypatch.setattr(modules, "resolve_slot_module", lambda slot, suffix="", m=module: SimpleNamespace(
            __name__="a_voice_module", __version__="1", open_voice=lambda **_options: m()))
        node = engine_module.tts_mod.build()
        node.load(lambda _message: None, voice="v", language="a")
        list(node.synth("Oh,", feeling="sad"))
        list(node.synth("Oh,", feeling="joy"))     # a short clip is kept by its feeling too: joy is not the sad one again
        list(node.synth("Oh,", feeling="joy"))
        assert asked == expect
