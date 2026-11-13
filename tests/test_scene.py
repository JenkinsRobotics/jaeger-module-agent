"""A scene: lines spoken by the agent's own voice, with moves timed to them."""

from __future__ import annotations

import numpy as np
import pytest

from jaeger_agent.core import scene
from jaeger_agent.core.engine import GemmaMultimodal

SCRIPT = """
# she waits, then asks
look left | hold 3
look center | emotion joy | say So, what did you do today?
say I spent mine learning to juggle. | action wave | on juggle
"""


class Chunk(np.ndarray):
    """A voice module's chunk: sound with its lip marks."""


def chunk(seconds: float, marks=(), rate: int = 100) -> Chunk:
    out = np.ones(int(seconds * rate), dtype=np.float32).view(Chunk)
    out.marks = [list(m) for m in marks]
    return out


def test_a_script_is_steps_in_order_and_a_move_rides_on_what_comes_next() -> None:
    assert scene.beats(SCRIPT) == [
        {"wait": 3.0, "cues": [{"look": "left", "hold": 3.0}]},                       # held three seconds, and the scene waits them out
        {"say": "So, what did you do today?", "cues": [{"look": "center", "emotion": "joy"}]},
        {"say": "I spent mine learning to juggle.", "cues": [{"action": "wave", "on": "juggle"}]},
    ]


@pytest.mark.parametrize("script, beats", [
    # the ways a small model wrote scenes when asked in words (2026-11-12): all on one line, a "wait 0", moves last
    ("look left | hold 3 | wait 0 | say How was your day?",
     [{"wait": 3.0, "cues": [{"look": "left", "hold": 3.0}]}, {"say": "How was your day?"}]),
    ("emotion excited | look center | say Hi there! | wait 1 | emotion joy | action spin | wait 1",
     [{"say": "Hi there!", "cues": [{"emotion": "excited", "look": "center"}]}, {"wait": 1.0},
      {"wait": 1.0, "cues": [{"emotion": "joy", "action": "spin"}]}]),
    ("emotion excited, action spin | say Watch this, everyone! | emotion joy, action bow | say Bye!",   # two moves in one step, by a comma
     [{"say": "Watch this, everyone!", "cues": [{"emotion": "excited", "action": "spin"}]},
      {"say": "Bye!", "cues": [{"emotion": "joy", "action": "bow"}]}]),
    ("say Goodbye | action bow",                                                        # a last move lands as the last line ends
     [{"say": "Goodbye"}, {"wait": scene.TAIL_S, "cues": [{"action": "bow"}]}]),
    ("emotion joy | say I spent mine learning to juggle. | action spin | on juggle",    # two moments on one line
     [{"say": "I spent mine learning to juggle.", "cues": [{"emotion": "joy"}, {"action": "spin", "on": "juggle"}]}]),
    ("look left | wait 2\\nsay Hi.",                                                     # line breaks written as the two characters \n
     [{"wait": 2.0, "cues": [{"look": "left"}]}, {"say": "Hi."}]),
    ([{"say": "Hi.", "emotion": "joy", "on": "Hi", "wait": 2}],                         # beats written out: a beat's moves ride on its own line
     [{"say": "Hi.", "wait": 2, "cues": [{"emotion": "joy", "on": "Hi"}]}]),
])
def test_scripts_as_they_are_written(script, beats) -> None:
    assert scene.beats(script) == beats


@pytest.mark.parametrize("script, complaint", [
    ("say Hello. | action bow | on goodbye", "Step 3: 'goodbye' is not in the line"),
    ("say Hi. | wait 90", "Step 2: wait is 0 to 30 seconds"),
    ("say Hi. | wait soon", "Step 2: wait is 0 to 30 seconds"),
    ("look left | hold 90", "Step 2: hold is 0 to 30 seconds"),
    ("say Hi. | on there", "Step 2: 'on' follows the move it places"),
    ("action bow | on there", "Step 2: 'on' needs a line before it"),
    ("say Hi. | bow", "Step 2: 'bow' needs a value, as in 'action bow'"),
    ("say Hi. | say", "Step 2: 'say' needs something after it"),
    ([{"say": "Hi."}, {}], "Beat 2: needs a line to say or a time to wait"),
    ([{"say": "Hi."}, "wave"], "Beat 2: a beat is a set of named parts"),
    ([{"say": "Hi.", "emotion": "joy", "on": "bye"}], "Beat 1: 'bye' is not in the line"),
    ([{"say": "Hi.", "wait": 90}], "Beat 1: wait is 0 to 30 seconds"),
    ("", "A scene needs at least one beat"),
])
def test_a_bad_beat_is_named_before_anything_is_said(script, complaint) -> None:
    with pytest.raises(ValueError, match=complaint):
        scene.beats(script)


def test_a_line_comes_whole_with_its_move_on_its_word() -> None:
    line, = scene.lines([{"say": "I spent mine learning to juggle.", "action": "wave", "on": "juggle"}])
    parts = [chunk(1.0, [(0.2, 0.3, "p")]), chunk(2.0, [(0.5, 0.6, "f")])]

    clip, = scene.voiced(line, lambda text: iter(parts), 100)

    assert line == "I spent mine learning to juggle."
    assert len(clip) == 300 and clip.dtype == np.float32
    assert clip.marks == [[0.2, 0.3, "p"], [1.5, 1.6, "f"]]       # the second chunk's marks, timed from the clip's start
    at = 3.0 * len("I spent mine learning to ") / len(line)        # by the share of the letters before the word
    assert clip.cues == [{"at": round(at, 3), "call": {"action": "wave"}}]


def test_a_line_the_voice_makes_no_sound_for_sends_no_empty_clip() -> None:
    line, = scene.lines([{"say": "...", "emotion": "joy"}])
    assert list(scene.voiced(line, lambda text: iter([]), 100)) == []      # a player cannot play nothing


def test_a_quiet_stretch_is_silent_sound_so_every_player_keeps_time() -> None:
    quiet, line, after = scene.lines([{"look": "left", "hold": 3, "wait": 1.5}, {"say": "Hi.", "emotion": "joy", "wait": 2}])

    clip, = scene.voiced(quiet, None, 100)
    assert quiet == "" and len(clip) == 150 and not clip.any()
    assert clip.cues == [{"at": 0, "call": {"look": "left", "hold": 3}}]   # a quiet beat's move lands as the quiet starts

    said, = scene.voiced(line, lambda text: iter([chunk(1.0)]), 100)
    assert said.cues == [{"at": 0, "call": {"emotion": "joy"}}]            # with no word named, as the line starts
    both, = scene.lines("action spin | say Now I juggle. | emotion joy | on juggle | look up | on Now")
    clip, = scene.voiced(both, lambda text: iter([chunk(1.3)]), 100)
    assert clip.cues == [{"at": 0, "call": {"action": "spin"}}, {"at": 0.0, "call": {"look": "up"}},
                         {"at": 0.6, "call": {"emotion": "joy"}}]         # several moments on one line, in the order they come
    rest, = scene.voiced(after, None, 100)
    assert after == "" and len(rest) == 200 and rest.cues == []            # the line's own quiet follows it, with nothing to do


class Runtime:
    def __init__(self) -> None:
        self.turns, self.notes = [], []

    def run_turn(self, text: str, *, session_key: str) -> str:
        self.turns.append(text)
        return "answer"

    def note(self, session_key: str, user_text: str, reply: str) -> None:
        self.notes.append((session_key, user_text, reply))

    def close(self) -> None:
        return None


class Voice:
    sample_rate = 100

    def synth(self, text: str):
        yield chunk(1.0, [(0.1, 0.2, "p")])


def test_the_engine_performs_a_scene_in_its_own_voice_and_remembers_its_lines() -> None:
    runtime, events = Runtime(), []
    engine = GemmaMultimodal(play=False, runtime=runtime, audio_mode="plain", output_mode="speech",
                             on_event=events.append)
    engine.node_tts = Voice()

    engine.perform(SCRIPT)

    sound = [e for e in events if e.kind == "audio"]
    assert [e.text for e in sound] == ["", "So, what did you do today?", "I spent mine learning to juggle."]
    assert [len(e.data) for e in sound] == [300, 100, 100]
    assert [e.data.cues for e in sound] == [
        [{"at": 0, "call": {"look": "left", "hold": 3.0}}],
        [{"at": 0, "call": {"look": "center", "emotion": "joy"}}],
        [{"at": round(len("I spent mine learning to ") / 32, 3), "call": {"action": "wave"}}],
    ]
    assert sound[1].data.marks == [[0.1, 0.2, "p"]]
    assert runtime.turns == []                                    # the model wrote none of it
    said = "So, what did you do today? I spent mine learning to juggle."
    assert [(e.text, e.data["source"]) for e in events if e.kind == "assistant"] == [(said, "scene")]
    assert runtime.notes == [(engine.node_llm.session_key, scene.DIRECTION, said)]   # so she knows what she asked
    states = [e.text for e in events if e.kind == "state"]
    assert states[0] == "speaking" and states[-1] == "listening"
    # the scene is fenced, so a player can hold its clips until all are voiced
    fence = [(e.kind, e.text) for e in events if e.kind in ("scene", "audio")]
    assert fence[0] == ("scene", "begin") and fence[-1] == ("scene", "end") and len(fence) == 5


def test_a_scene_staged_inside_a_turn_is_performed_once_the_reply_is_spoken() -> None:
    class Acting(Runtime):
        def run_turn(self, text: str, *, session_key: str) -> str:
            self.turns.append(text)
            if "act" in text:                                     # what the model's scene tool does mid-turn
                assert self.engine.stage("emotion joy | say Ta da!") == 1
            return "" if "silently" in text else "Watch this."

    runtime, events = Acting(), []
    engine = runtime.engine = GemmaMultimodal(play=False, runtime=runtime, audio_mode="plain", output_mode="speech",
                                              on_event=events.append)
    engine.node_tts = Voice()

    engine.send_text("act it out")

    order = [(e.kind, e.text) for e in events if e.kind in ("assistant", "audio", "scene") or (e.kind, e.text) == ("state", "listening")]
    assert order == [("assistant", "Watch this."), ("audio", "Watch this."), ("scene", "begin"), ("audio", "Ta da!"),
                     ("scene", "end"), ("state", "listening")]
    assert [e.data.cues for e in events if e.kind == "audio" and e.text == "Ta da!"] == [[{"at": 0, "call": {"emotion": "joy"}}]]
    assert runtime.notes == []                                    # the model's own call already holds the script

    events.clear()
    engine.send_text("act it out silently")                       # the model said nothing itself: the scene is the whole reply
    assert [(e.kind, e.text) for e in events if e.kind in ("audio", "scene")] == [("scene", "begin"), ("audio", "Ta da!"), ("scene", "end")]

    events.clear()
    engine.send_text("hello")                                     # nothing staged: nothing more is performed
    assert [e.text for e in events if e.kind == "audio"] == ["Watch this."]
    with pytest.raises(ValueError, match="Step 3"):
        engine.stage("say Hello. | action bow | on goodbye")                   # a bad script is refused to the tool, and stages nothing
    engine.send_text("hello")
    assert not [e for e in events if e.kind == "scene"]


def test_a_bad_scene_says_nothing() -> None:
    events = []
    engine = GemmaMultimodal(play=False, runtime=Runtime(), audio_mode="plain", on_event=events.append)
    engine.node_tts = Voice()

    with pytest.raises(ValueError, match="Step 3"):
        engine.perform("say Hello. | action bow | on goodbye")

    assert events == []


def test_the_body_fits_each_call_before_a_word_and_may_refuse_the_scene() -> None:
    asked = []

    def strict(call: dict) -> dict:                               # a director is told, and fixes the script
        asked.append(call)
        if call.get("action") == "wave":
            raise ValueError("This body has no wave.")
        return call

    events = []
    engine = GemmaMultimodal(play=False, runtime=Runtime(), audio_mode="plain", output_mode="speech", on_event=events.append)
    engine.node_tts = Voice()

    with pytest.raises(ValueError, match="This body has no wave."):
        engine.perform("emotion joy | hold 2 | say Hi there. | action wave | on there", fit=strict)
    assert events == []                                           # before the scene starts
    assert asked == [{"emotion": "joy", "hold": 2.0}, {"action": "wave"}]   # each call as the body's tool will get it

    with pytest.raises(ValueError, match="This body has no wave."):
        engine.stage("say Hi. | action wave", fit=strict)
    engine.send_text("hello")
    assert not [e for e in events if e.kind == "scene"]           # nothing was staged


def test_a_call_the_body_fits_plays_without_what_it_cannot_do() -> None:
    def lenient(call: dict) -> dict:                              # the model's scene still plays, less what the body lacks
        return {k: v for k, v in call.items() if (k, v) not in (("action", "wave"), ("emotion", "happy"))}

    quiet, line = scene.lines("emotion happy | action wave | hold 1 | emotion joy | say Hi there. | action wave | on there"
                              " | emotion happy | look up | on Hi", fit=lenient)
    assert quiet.wait == 1.0 and quiet.cues == []                 # nothing left of that call but its hold: no cue, the quiet stays
    assert line == "Hi there." and line.cues == [{"emotion": "joy"}, {"look": "up", "on": "Hi"}]   # a kept move keeps its word
