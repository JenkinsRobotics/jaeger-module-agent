"""The agent's voice and hearing are whichever Jaeger module fills the slot.

The agent carries no loader of its own and no fallback copy: an upgrade to a
module reaches the agent, and a missing module is a clear error.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from jaeger_agent.core import engine


def fill_slots(monkeypatch, **packages):
    """Stand-in for JaegerOS slot discovery: slot name -> the module filling it."""
    import jaeger_os.core.modules as modules

    monkeypatch.setattr(modules, "resolve_slot_module",
                        lambda slot, suffix="": packages.get(slot))


def test_voice_node_speaks_through_the_module_in_the_tts_slot(monkeypatch):
    opened = []

    class Voice:
        compute = "mlx"

        def synth(self, text):
            yield np.asarray([0.5], dtype=np.float32)
            yield np.asarray([-0.5], dtype=np.float32)

    def open_voice(*, voice, language):
        opened.append((voice, language))
        return Voice()

    fill_slots(monkeypatch, tts=SimpleNamespace(
        __name__="some_voice_module", __version__="9.9", open_voice=open_voice))
    node = engine.tts_mod.build()
    node.load(lambda _message: None, voice="v", language="a")

    assert opened == [("v", "a")]
    assert [chunk.tolist() for chunk in node.synth("hi")] == [[0.5], [-0.5]]
    assert node.info == {"module": "some_voice_module", "version": "9.9",
                         "model": "v", "compute": "mlx"}


def test_voice_node_keeps_short_clips_once_made(monkeypatch):
    made = []

    class Voice:
        def synth(self, text):
            made.append(text)
            yield np.asarray([0.5], dtype=np.float32)
            yield np.asarray([-0.5], dtype=np.float32)

    fill_slots(monkeypatch, tts=SimpleNamespace(
        __name__="m", open_voice=lambda **_options: Voice()))
    node = engine.tts_mod.build()
    node.load(lambda _message: None, voice="v", language="a")

    first = [chunk.tolist() for chunk in node.synth("Oh,")]
    assert [chunk.tolist() for chunk in node.synth("Oh,")] == first == [[0.5], [-0.5]]
    assert made == ["Oh,"]                       # the second one was not made again

    long = "A whole sentence."
    list(node.synth(long)), list(node.synth(long))
    assert made == ["Oh,", long, long]           # only short clips are kept

    next(node.synth("Hmm?"))                     # cut off after its first chunk
    assert len(list(node.synth("Hmm?"))) == 2    # half a clip is never kept

    node.load(lambda _message: None, voice="other", language="a")
    list(node.synth("Oh,"))
    assert made[-1] == "Oh,"                     # another voice makes its own


def test_the_voice_changes_without_loading_again_and_lip_marks_come_with_the_sound(monkeypatch):
    made, opened = [], []

    class Marked(np.ndarray):
        marks: list = []

    class Voice:
        def synth(self, text, *, voice="v"):
            made.append((text, voice))
            chunk = np.asarray([0.5], dtype=np.float32).view(Marked)
            chunk.marks = [[0.0, 0.05, "p"]]            # the module times the lips; the node only carries them
            yield chunk

    fill_slots(monkeypatch, tts=SimpleNamespace(
        __name__="m", open_voice=lambda **options: opened.append(options) or Voice()))
    node = engine.tts_mod.build()
    node.load(lambda _message: None, voice="v", language="a")

    list(node.synth("Oh,"))
    node.set_voice("bf_emma")                           # another character speaks now
    (chunk,) = node.synth("Oh,")
    (kept,) = node.synth("Oh,")
    list(node.synth("Oh,", voice="am_adam"))            # one line in a borrowed voice
    list(node.synth("Oh,"))
    assert made == [("Oh,", "v"), ("Oh,", "bf_emma"), ("Oh,", "am_adam")]   # each voice keeps its own short clips
    assert chunk.marks == kept.marks == [[0.0, 0.05, "p"]]
    assert len(opened) == 1 and node.info["model"] == "bf_emma"            # the module was not opened again


def test_hearing_node_hears_through_the_module_in_the_stt_slot(monkeypatch):
    opened = []
    node = engine.stt_mod.build()

    class Hearing:
        compute = "metal"

        def transcribe(self, pcm, *, language="en", prompt=None, abort=None):
            assert node._lock.locked()      # preview and final share one model
            return [SimpleNamespace(text="Hello"), SimpleNamespace(text="there.")]

    def open_hearing(*, model):
        opened.append(model)
        return Hearing()

    fill_slots(monkeypatch, stt=SimpleNamespace(
        __name__="some_hearing_module", __version__="1.2", open_hearing=open_hearing))
    node.load(lambda _message: None, model_name="m")

    assert opened == ["m"]
    assert node.transcribe(np.zeros(160, dtype=np.float32)) == "Hello there."
    assert node.info == {"module": "some_hearing_module", "version": "1.2",
                         "model": "m", "compute": "metal"}


def test_nothing_in_the_slot_is_a_clear_error_not_a_fallback(monkeypatch):
    fill_slots(monkeypatch)
    with pytest.raises(RuntimeError, match=r"'tts' slot.*jaeger-agent\[multimodal\]"):
        engine.tts_mod.build().load(lambda _message: None)
    with pytest.raises(RuntimeError, match=r"'stt' slot.*jaeger-agent\[multimodal\]"):
        engine.stt_mod.build().load(lambda _message: None)


def test_a_voice_module_that_exits_during_load_is_a_runtime_failure(monkeypatch):
    # A language-asset downloader calling sys.exit would otherwise end the
    # host's boot thread without reporting anything.
    def open_voice(**_options):
        raise SystemExit(1)

    fill_slots(monkeypatch, tts=SimpleNamespace(__name__="m", open_voice=open_voice))
    with pytest.raises(RuntimeError, match="could not be initialized"):
        engine.tts_mod.build().load(lambda _message: None)


def test_streaming_decode_uses_the_neutral_hearing_call():
    calls = []
    cancel = threading.Event()

    class Hearing:
        def transcribe(self, pcm, *, language="en", prompt=None, abort=None):
            calls.append((len(pcm), language, prompt, abort))
            if len(calls) == 1:
                return [SimpleNamespace(text="Hello there.", start_ms=0, end_ms=1000),
                        SimpleNamespace(text="How are", start_ms=1000, end_ms=1900)]
            return [SimpleNamespace(text="How are", start_ms=0, end_ms=900)]

    stream = engine.IncrementalTranscriber(Hearing())
    stream.feed(np.zeros(16000 * 3, dtype=np.float32))

    assert stream.decode_pass(abort_cb=cancel.is_set) == ["Hello there."]
    # 3 s of audio + 250 ms pad went in; the committed sentence ended at
    # 1000 ms, so exactly its second of audio is cut from the buffer ...
    assert calls == [(16000 * 3 + 4000, "en", None, cancel.is_set)]
    assert stream.buffer_seconds == 2.0
    # ... and its words become the next decode's prompt.
    stream.decode_pass()
    assert calls[-1] == (16000 * 2 + 4000, "en", "Hello there.", None)


def test_listen_hears_through_the_same_slot_with_the_agents_one_default(monkeypatch):
    import importlib

    from jaeger_agent.core.policy import STT_MODEL

    listen = importlib.import_module("jaeger_agent.tools.listen")   # not the tool function

    opened = []

    def open_hearing(*, model):
        opened.append(model)
        return "the hearing"

    fill_slots(monkeypatch, stt=SimpleNamespace(__name__="m", open_hearing=open_hearing))
    monkeypatch.setattr(listen, "_cached_model", None)

    assert listen._get_model(listen._DEFAULT_MODEL) == "the hearing"
    assert opened == [STT_MODEL]


def test_the_listen_tool_and_the_voice_loop_share_one_hearing(monkeypatch):
    import importlib
    import sys

    from jaeger_agent.core import parts
    from jaeger_agent.core.policy import STT_MODEL

    listen = importlib.import_module("jaeger_agent.tools.listen")   # not the tool function
    opened = []

    class Hearing:
        compute = "metal"

        def transcribe(self, pcm, **_options):
            assert parts.HEARING_LOCK.locked()      # one pass at a time, whoever asks
            return [SimpleNamespace(text="hi")]

    def open_hearing(*, model):
        opened.append(model)
        return Hearing()

    fill_slots(monkeypatch, stt=SimpleNamespace(__name__="m", open_hearing=open_hearing))
    monkeypatch.setattr(listen, "_cached_model", None)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(listen, "_record_via_avaudio",
                        lambda seconds, np_: np.zeros(16000, dtype=np.float32))
    node = engine.stt_mod.build()
    node.load(lambda _message: None)

    assert listen._get_model(listen._DEFAULT_MODEL) is node.model
    assert opened == [STT_MODEL]                    # one copy of the model in memory
    assert listen.listen(seconds=1)["transcript"] == "hi"


class _Runtime:
    def run_turn(self, text: str, *, session_key: str) -> str:
        return "ok"

    def close(self) -> None:
        return None


def _engine_with_fake_parts(monkeypatch, *, hearing_error=None):
    hearing_loading = threading.Event()
    loaded_on: dict[str, threading.Thread] = {}

    class Vision:
        ready = False

        def load(self, *_a, **_k):
            return None

    class Model:
        def load(self, *_a, **_k):
            assert hearing_loading.wait(2.0), "hearing waited for the model"

    class Voice:
        tts = "the voice"

        def load(self, say, **_k):
            loaded_on["voice"] = threading.current_thread()

    class Hearing:
        model = "the hearing"

        def load(self, say, **_k):
            loaded_on["hearing"] = threading.current_thread()
            hearing_loading.set()
            if hearing_error:
                raise RuntimeError(hearing_error)

    monkeypatch.setattr(engine, "SileroVad", lambda _path: object())
    eng = engine.GemmaMultimodal(play=False, runtime=_Runtime(), audio_mode="plain")
    eng.node_vision, eng.node_llm = Vision(), Model()
    eng.node_tts, eng.node_stt = Voice(), Hearing()
    eng.loaded_on = loaded_on
    return eng


def test_hearing_loads_beside_the_model_and_the_voice_on_the_callers_thread(monkeypatch):
    eng = _engine_with_fake_parts(monkeypatch)
    eng.load()
    try:
        assert (eng.tts, eng.stt) == ("the voice", "the hearing")
        # An MLX voice loaded off the caller's thread crashes the process at exit.
        assert eng.loaded_on["voice"] is threading.current_thread()
        assert eng.loaded_on["hearing"] is not threading.current_thread()
    finally:
        eng.close()


def test_a_part_that_fails_to_load_fails_the_start_up(monkeypatch):
    eng = _engine_with_fake_parts(monkeypatch, hearing_error="no hearing here")
    with pytest.raises(RuntimeError, match="no hearing here"):
        eng.load()
    eng.close()
