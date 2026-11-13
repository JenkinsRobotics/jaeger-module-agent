"""The voice starts on the first finished sentence, not on the finished reply."""

from __future__ import annotations

import threading

import numpy as np

from jaeger_agent import Message, ProviderAdapter
from jaeger_agent.core.engine import GemmaMultimodal
from jaeger_agent.core.outputs import StreamedSpeech
from jaeger_agent.core.runtime import DefaultAgentRuntime


def _fed(stream: StreamedSpeech, *pieces: str) -> list[str]:
    return [sentence for piece in pieces for sentence in stream.feed(piece)]


def test_sentences_come_out_as_they_finish_and_rest_is_the_tail() -> None:
    stream = StreamedSpeech(mode="speech", input_modality="text")
    assert _fed(stream, "Octopuses have ", "three hearts. They ") == [
        "Octopuses have three hearts."
    ]
    assert _fed(stream, "also have blue blood.\nAnd nine") == ["They also have blue blood."]
    assert stream.rest(
        "Octopuses have three hearts. They also have blue blood.\nAnd nine brains."
    ) == "And nine brains."


def test_dynamic_mode_waits_for_the_models_channel_choice() -> None:
    stream = StreamedSpeech(mode="dynamic", input_modality="text")
    assert _fed(stream, "[", "OUTPUT", ":SPE") == []
    assert _fed(stream, "ECH]", "It is noon now. ") == ["It is noon now."]

    text_only = StreamedSpeech(mode="dynamic", input_modality="speech")
    assert _fed(text_only, "[OUTPUT:TEXT] Here is the table. Read it. ") == []

    typed = StreamedSpeech(mode="dynamic", input_modality="text")
    assert _fed(typed, "No directive here. Still none. ") == []

    spoken = StreamedSpeech(mode="dynamic", input_modality="speech")
    assert _fed(spoken, "No directive here. Still none. ") == [
        "No directive here.", "Still none."
    ]


def test_one_word_fragments_join_the_next_sentence() -> None:
    stream = StreamedSpeech(mode="speech", input_modality="speech")
    assert _fed(stream, "Sure. Dr. Smith is in. 1. First point. ") == [
        "Sure. Dr. Smith is in.", "1. First point."
    ]


def test_a_one_word_exclamation_opens_the_reply_as_its_own_breath() -> None:
    # as Jaeger Avatar's old path does: "Oh!" is voiced while the rest is still being written
    stream = StreamedSpeech(mode="speech", input_modality="speech")
    assert _fed(stream, "Oh! You made ") == ["Oh!"]
    # only the opening: later ones join what follows
    assert _fed(stream, "that yourself? Wow! That is lovely. ") == [
        "You made that yourself?", "Wow! That is lovely."
    ]

    after_a_call = StreamedSpeech(mode="speech", input_modality="text")
    assert _fed(after_a_call, "<|tool_call>call:avatar{}<tool_call|>", "Hmm? The time, ") == ["Hmm?"]


def test_a_directive_or_tool_call_never_reaches_the_voice() -> None:
    stream = StreamedSpeech(mode="dynamic", input_modality="speech")
    said = _fed(
        stream,
        "[OUTPUT:SPEECH] Let me check that.",
        '<tool_call>{"name": "clock", "arguments": {}}</tool_call>',
        "[OUTPUT:SPEECH] It is five. Done",
    )
    assert said == ["Let me check that.", "It is five."]
    assert stream.rest("It is five. Done") == "Done"


def test_the_voice_comes_back_after_a_tool_call_without_a_directive() -> None:
    stream = StreamedSpeech(mode="speech", input_modality="text")
    said = _fed(
        stream,
        '<|tool_call>call:avatar{emotion:<|"|>joy<|"|>}<tool_',
        "call|>",
        "Happy to see you! It has been ",
        "a while. ",
    )
    assert said == ["Happy to see you!", "It has been a while."]

    stray = StreamedSpeech(mode="speech", input_modality="text")
    assert _fed(stray, "Is 3 < 5 here? ", "More words follow. ") == ["Is 3"]   # a lone "<" still mutes the rest


def test_a_made_up_tag_is_dropped_and_the_voice_keeps_going() -> None:
    """A small model sometimes writes its feelings as tags of its own
    (measured 2026-11-12 in Jaeger Avatar: "<emotion:joy> <action:wave>
    </emotion> Hey there!" on every turn of one conversation). Muting the rest
    held the whole reply back until it was finished — first sound 1.35-2.6 s
    against a reply ready at 0.75-1.15 s — and the tags were then read aloud."""
    stream = StreamedSpeech(mode="speech", input_modality="text")
    said = _fed(stream, "<emotion:joy> <action:wave> </emotion> Hey there! ", "<b>So</b> nice to ", "meet you. ")
    assert said == ["Hey there!", "So nice to meet you."]
    assert stream.rest("<emotion:joy> <action:wave> </emotion> Hey there! <b>So</b> nice to meet you. Bye now.") == "Bye now."

    split = StreamedSpeech(mode="speech", input_modality="text")
    assert _fed(split, "<emo", "tion:joy", "> Hi there! ") == ["Hi there!"]      # a tag that arrives in pieces

    call = StreamedSpeech(mode="speech", input_modality="text")                  # a real call is still never spoken
    assert _fed(call, '<tool_call>{"name": "clock"}</tool_call>', "<call:avatar{}> ", "It is five now. ") == ["It is five now."]


def test_a_made_up_tag_is_not_read_aloud_when_the_reply_is_spoken_whole() -> None:
    from jaeger_agent.core.outputs import decide_output

    spoken = decide_output(mode="speech", input_modality="text",
                           reply="<emotion:joy> <action:wave> </emotion> Hey there!")
    assert spoken.speech_text == "Hey there!"


def test_the_voice_comes_back_after_a_bare_call_in_the_middle_of_the_words() -> None:
    stream = StreamedSpeech(mode="speech", input_modality="text")
    said = _fed(
        stream,
        'A puppy, wow! <call:avatar{emotion:<|"|>joy<|"|>}',
        "> You must be so excited! ",
        "Tell me its name. ",
    )
    assert said == ["A puppy, wow!", "You must be so excited!", "Tell me its name."]


def test_rest_is_the_whole_reply_when_nothing_was_spoken() -> None:
    stream = StreamedSpeech(mode="dynamic", input_modality="speech")
    assert _fed(stream, "[SILENT]\n") == []
    assert stream.rest("All of it. Every word.") == "All of it. Every word."


class _FakeVoice:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def synth(self, text: str):
        self.texts.append(text)
        yield np.zeros(240, dtype=np.float32)


class _StreamingRuntime:
    """Hands its reply out in pieces, the way the built-in runtime does."""

    def __init__(self, voice: _FakeVoice) -> None:
        self.voice = voice
        self.heard_before_reply_ended: list[str] = []

    def run_turn(self, text: str, *, session_key: str, on_delta=None) -> str:
        if on_delta is not None:
            on_delta("First sentence here. ")
            on_delta("Second one")
            for _ in range(200):  # the voice runs beside the model, give it a moment
                if self.voice.texts:
                    break
                threading.Event().wait(0.01)
        self.heard_before_reply_ended = list(self.voice.texts)
        return "First sentence here. Second one ends now."

    def close(self) -> None:
        return None


def _engine(runtime, voice, events, **kw):
    engine = GemmaMultimodal(play=False, runtime=runtime, audio_mode="plain",
                             on_event=events.append, **kw)
    engine.node_tts = voice
    return engine


def test_engine_speaks_the_first_sentence_while_the_reply_is_still_coming() -> None:
    voice, events = _FakeVoice(), []
    runtime = _StreamingRuntime(voice)
    engine = _engine(runtime, voice, events, output_mode="speech")
    engine.send_text("hello")
    assert runtime.heard_before_reply_ended == ["First sentence here."]
    assert voice.texts == ["First sentence here.", "Second one ends now."]
    assistant = [e.text for e in events if e.kind == "assistant"]
    assert assistant == ["First sentence here. Second one ends now."]
    # the turn's own clock says where the time went: model → first words → voice
    notes = [e.text for e in events if e.kind == "latency"]
    assert notes[:3] == ["First token", "First words", "First audio"]


def test_text_only_mode_never_streams_to_the_voice() -> None:
    voice, events = _FakeVoice(), []
    engine = _engine(_StreamingRuntime(voice), voice, events, output_mode="text")
    engine.send_text("hello")
    assert voice.texts == []


def test_force_listen_stops_a_reply_that_is_already_being_spoken() -> None:
    voice, events = _FakeVoice(), []

    class Interrupted(_StreamingRuntime):
        def run_turn(self, text, *, session_key, on_delta=None):
            reply = super().run_turn(text, session_key=session_key, on_delta=on_delta)
            engine.force_listen()
            return reply

    engine = _engine(Interrupted(voice), voice, events, output_mode="speech")
    engine.send_text("hello")
    assert voice.texts == ["First sentence here."]
    assert not engine.speaking.is_set()
    assert [e.text for e in events if e.kind == "state"][-1] == "listening"


def test_built_in_runtime_hands_out_the_pieces_for_one_turn_only() -> None:
    class PiecewiseAdapter(ProviderAdapter):
        name = "piecewise"

        def format_messages(self, messages, tools, system):
            return messages

        def call(self, formatted, interrupt_event, **kwargs):
            self.on_delta = kwargs.get("on_delta")
            if self.on_delta is not None:
                self.on_delta("Hello ")
                self.on_delta("there.")
            return formatted

        def parse_response(self, raw) -> Message:
            return Message(role="assistant", content="Hello there.")

        def supports(self, feature: str) -> bool:
            return False

    adapter = PiecewiseAdapter()
    runtime = DefaultAgentRuntime(config={"tools_enabled": False}, adapter=adapter)
    pieces: list[str] = []
    result = runtime.run_turn("hi", session_key="s", on_delta=pieces.append)
    assert result.text == "Hello there."
    assert pieces == ["Hello ", "there."]
    runtime.run_turn("again", session_key="s")
    assert adapter.on_delta is None


class _FakeEars:
    def transcribe(self, audio) -> str:
        return "what is the capital of France"


def _heard_without_a_wake_phrase(**config) -> list[str]:
    from jaeger_agent.core.config import MultimodalConfig

    voice, events = _FakeVoice(), []
    engine = _engine(_StreamingRuntime(voice), voice, events, output_mode="speech",
                     config=MultimodalConfig(audio_mode="plain", **config))
    engine.node_stt = _FakeEars()
    engine._on_turn(np.zeros(16000, dtype=np.float32))
    return [e.kind for e in events]


def test_speech_without_the_wake_phrase_is_only_overheard_by_default() -> None:
    kinds = _heard_without_a_wake_phrase()
    assert "overheard" in kinds and "assistant" not in kinds


def test_an_open_microphone_app_can_turn_the_wake_phrase_off() -> None:
    kinds = _heard_without_a_wake_phrase(wake_required=False)
    assert "overheard" not in kinds and "assistant" in kinds


def test_an_app_can_switch_character_and_each_keeps_its_own_conversation() -> None:
    voice, events = _FakeVoice(), []

    class Recording(_StreamingRuntime):
        warmed: list = []
        turns: list = []

        def warmup(self, *, session_key, system_prompt=""):
            self.warmed.append((session_key, system_prompt))

        def run_multimodal_turn(self, content, *, text, system_prompt="",
                                session_key, on_delta=None):
            self.turns.append((session_key, system_prompt))
            return "Fine."

    runtime = Recording(voice)
    engine = _engine(runtime, voice, events, output_mode="speech",
                     extra_prompt="You are Nori.")
    engine.switch_session("mochi", "You are Mochi.")
    assert runtime.warmed[-1][0] == "mochi"
    assert "You are Mochi." in runtime.warmed[-1][1]
    engine.send_text("hello")
    key, system = runtime.turns[-1]
    assert key == "mochi"
    assert "You are Mochi." in system and "Nori" not in system


class _CharacterVoice(_FakeVoice):
    sample_rate = 22050
    info = {"module": "m", "version": "1", "compute": "mlx", "model": "v"}

    def __init__(self) -> None:
        super().__init__()
        self.voices: list[str] = []

    def set_voice(self, voice: str) -> None:
        self.voices.append(voice)

    def synth(self, text: str, *, voice=None):
        self.texts.append((text, voice))
        yield np.zeros(240, dtype=np.float32)


class _QuietRuntime(_StreamingRuntime):
    def warmup(self, *, session_key, system_prompt=""):
        return None


def test_a_character_brings_its_voice_and_a_line_can_be_spoken_without_the_model() -> None:
    voice, events = _CharacterVoice(), []
    engine = _engine(_QuietRuntime(voice), voice, events, output_mode="speech")
    engine.switch_session("mochi", "You are Mochi.", voice="af_bella")
    engine.switch_session("vita", "You are Vita.")          # no voice named: the voice stays as it is
    assert voice.voices == ["af_bella"]

    assert [len(chunk) for chunk in engine.synth("Hello.", voice="bm_george")] == [240]
    list(engine.synth("Hello."))
    assert voice.texts == [("Hello.", "bm_george"), ("Hello.", None)]
    assert not [e for e in events if e.kind in ("assistant", "audio", "user")]   # the voice alone: no turn
    assert engine.sample_rate == 22050


def test_an_app_asks_the_engine_for_the_call_form_and_what_each_part_runs_on() -> None:
    from types import SimpleNamespace

    voice = _CharacterVoice()
    runtime = _QuietRuntime(voice)
    runtime.adapter = SimpleNamespace(tool_call_example=lambda name, args: f"<call:{name}{args}>")
    runtime.health = lambda: {"model": "gemma-test"}
    runtime.config = SimpleNamespace(ctx=32768, tools_enabled=True, max_iterations=24)
    engine = _engine(runtime, voice, [], output_mode="speech")
    engine.node_stt = SimpleNamespace(info={"module": "h", "compute": "metal"})

    assert engine.call_example("avatar", {"emotion": "joy"}) == "<call:avatar{'emotion': 'joy'}>"
    assert engine.report() == {
        "voice": voice.info, "hearing": {"module": "h", "compute": "metal"}, "agent": {"model": "gemma-test"},
        "ctx": 32768, "tools_enabled": True, "max_iterations": 24}
    runtime.adapter = object()                              # a model whose call form is not known
    assert engine.call_example("avatar", {}) == ""


def test_rest_lines_up_by_words_when_a_short_opener_joined_the_next_call() -> None:
    stream = StreamedSpeech(mode="speech", input_modality="text")
    # "Sure." came from the model call before a tool ran, the rest from the one after
    assert _fed(stream, "Sure.", "\n", "It is noon now. ") == ["Sure.\nIt is noon now."]
    assert stream.rest("It is noon now. Anything else?") == "Anything else?"


def test_the_first_words_go_out_at_the_first_clause_so_the_voice_starts_sooner() -> None:
    stream = StreamedSpeech(mode="speech", input_modality="text")
    assert _fed(stream, "Oh wow, octopuses ") == ["Oh wow,"]
    # only the opening: after it, whole sentences
    assert _fed(stream, "are clever, truly. They ") == ["octopuses are clever, truly."]
    assert stream.rest("Oh wow, octopuses are clever, truly. They hide.") == "They hide."

    opener = StreamedSpeech(mode="speech", input_modality="text")
    assert _fed(opener, "Sure. It is so good to meet you; I am ") == [
        "Sure. It is so good to meet you;"
    ]

    # a one-word opening clause is a breath of its own: short clips are kept once made, so it is heard at once
    one_word = StreamedSpeech(mode="speech", input_modality="text")
    assert _fed(one_word, "Oh, I love ") == ["Oh,"]
    assert _fed(one_word, "stories, truly! Next ") == ["I love stories, truly!"]

    counting = StreamedSpeech(mode="speech", input_modality="text")
    assert _fed(counting, "1, 2, 3 and ") == []  # a count is not cut at its first number

    after_a_call = StreamedSpeech(mode="speech", input_modality="text")
    assert _fed(after_a_call, '<|tool_call>call:avatar{}<tool_call|>', "So happy, truly ") == ["So happy,"]



def test_a_long_opening_with_no_comma_goes_out_at_a_breath_point() -> None:
    stream = StreamedSpeech(mode="speech", input_modality="text")
    assert _fed(stream, "I'm feeling super", " energetic", " and") == []  # "and" may still become "android"
    assert _fed(stream, " ready ") == ["I'm feeling super energetic"]
    # only the opening: the rest of the sentence follows whole
    assert _fed(stream, "to chat with you and your friends! How ") == ["and ready to chat with you and your friends!"]
    assert stream.rest("I'm feeling super energetic and ready to chat with you and your friends! How about you?") == "How about you?"

    few = StreamedSpeech(mode="speech", input_modality="text")
    assert _fed(few, "You and I should ") == []  # too few words before the joining word for a breath
    assert _fed(few, "go dancing because ") == ["You and I should go dancing"]

    inside_a_word = StreamedSpeech(mode="speech", input_modality="text")
    assert _fed(inside_a_word, "I really do love androids whenever ") == []

    later = StreamedSpeech(mode="speech", input_modality="text")
    assert _fed(later, "Hi there! I really do love cats and ") == ["Hi there!"]

    counting = StreamedSpeech(mode="speech", input_modality="text")
    assert _fed(counting, "1, 2, 3 are the first numbers and ") == []  # a count stays whole, as before

    unheard = StreamedSpeech(mode="dynamic", input_modality="text")
    assert _fed(unheard, "I'm feeling super energetic and ready ") == []
