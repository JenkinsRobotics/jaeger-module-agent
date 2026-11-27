"""Her manner is said with each turn, not in the system prompt.

How a character talks right now (formal, few words, no jokes: a pack's dials,
or ones someone has just turned) rides on what the user said, the way the
``[context:]`` line does. Measured on gemma-4-E4B, 2026-11-22, same asks, same
character sheet: said in the system prompt, three rules at once changed almost
nothing (4 contractions in five replies stayed 3 to 4) and every change cost
a new conversation and 11 to 14 s before her next word, the whole prompt
being read again. Said with the turn they were followed (23 words a reply
became 7; contractions and exclamation marks 0), they changed in the middle
of a conversation, and the avatar tool was still called on 48 of 48 turns.
"""
from __future__ import annotations

import numpy as np

from jaeger_agent.core.config import MultimodalConfig
from jaeger_agent.core.engine import GemmaMultimodal


class _Voice:
    def synth(self, text: str):
        yield np.zeros(240, dtype=np.float32)


class _Runtime:
    def __init__(self) -> None:
        self.asked: list[str] = []

    def run_turn(self, text: str, *, session_key: str) -> str:
        self.asked.append(text)
        return "Fine."

    def note(self, session_key: str, user_text: str, reply: str) -> None:
        return None

    def close(self) -> None:
        return None


def test_her_manner_rides_on_each_turn_and_changes_between_turns() -> None:
    runtime, events = _Runtime(), []
    engine = GemmaMultimodal(play=False, runtime=runtime, audio_mode="plain", output_mode="speech", extra_prompt="You are Vita.",
                             on_event=events.append, config=MultimodalConfig(wake_required=False))
    engine.node_tts = _Voice()

    engine.send_text("Hello.")
    assert runtime.asked == ["Hello."]                       # no manner: the turn as it was

    engine.manner = "Speak formally. At most ten words."
    before = engine._system_prompt()
    engine.send_text("Hello again.")
    assert runtime.asked[-1] == "[manner: Speak formally. At most ten words.]\nHello again."
    assert engine._system_prompt() == before and "formally" not in before    # the prompt she was loaded with is untouched: nothing is read again
    assert [e.text for e in events if e.kind == "user"] == ["Hello.", "Hello again."]   # what they said is shown as they said it

    engine.manner = ""
    engine.send_text("And now?")
    assert runtime.asked[-1] == "And now?"


class _LaughingVoice(_Voice):
    def sounds(self) -> list:
        return ["giggle", "laugh"]


def test_a_voice_with_real_laughs_tells_the_model_how_to_write_one_and_one_without_says_nothing() -> None:
    engine = GemmaMultimodal(play=False, runtime=_Runtime(), audio_mode="plain", output_mode="speech",
                             config=MultimodalConfig(wake_required=False))
    engine.node_tts = _Voice()
    assert "[laugh]" not in engine._system_prompt()
    engine.node_tts = _LaughingVoice()
    assert "[giggle], [laugh]" in engine._system_prompt()
