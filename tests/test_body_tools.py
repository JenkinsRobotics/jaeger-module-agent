"""The agent's hands on a body: the avatar tool and the scene tool (jaeger_agent/body/tools.py).

Came with the tools from JaegerAvatar (tests/test_agent_tools.py), November 22, 2026. There they drove a real control
link; here the body is a stand-in that takes what it offers and refuses the rest, as a body's control does.
"""
from __future__ import annotations

from jaeger_agent.body.tools import HOLD_MS, avatar_tool, fitted, scene_tool

OFFERS = {"emotions": ["neutral", "joy"], "reactions": ["wink"], "actions": ["clap"]}
KIND = {"emotion": "emotions", "reaction": "reactions", "act": "actions"}


class Stage:
    """A body's control: send(op, value, ttl_ms)."""

    def __init__(self, on: bool = True) -> None:
        self.on, self.state, self.held = on, {}, {}

    def __call__(self, op, value, ttl_ms):
        if not self.on:
            return {"ok": False, "error": "No avatar attached"}
        if op == "offers":
            return {"ok": True, "offers": OFFERS}
        if op in KIND and value not in OFFERS[KIND[op]]:
            return {"ok": False, "error": f"Unknown {'action' if op == 'act' else op}", "offers": OFFERS}
        self.state[op], self.held[op] = value, ttl_ms
        return {"ok": True}


def test_an_emotion_reaches_the_body_and_is_held() -> None:
    stage = Stage()
    assert avatar_tool(stage)(emotion="joy") == {"ok": True, "did": ["emotion joy"]}
    assert (stage.state["emotion"], stage.held["emotion"]) == ("joy", HOLD_MS)


def test_a_feeling_this_body_lacks_comes_back_with_what_it_offers() -> None:
    out = avatar_tool(Stage())(emotion="smug")
    assert not out["ok"] and "Unknown emotion" in out["errors"][0]
    assert out["offers"] == OFFERS


def test_an_action_may_be_a_gesture_or_a_reaction() -> None:
    stage = Stage()
    avatar = avatar_tool(stage)
    assert avatar(action="clap")["did"] == ["action clap"]
    assert avatar(action="wink")["did"] == ["action wink"]
    assert (stage.state["act"], stage.state["reaction"]) == ("clap", "wink")


def test_look_words_become_a_gaze() -> None:
    stage = Stage()
    avatar = avatar_tool(stage)
    assert avatar(look="up left")["ok"] and stage.state["gaze"] == [-1, 1]
    assert "look" in avatar(look="sideways")["errors"][0]


def test_one_call_can_feel_and_move_and_reports_each_part() -> None:
    out = avatar_tool(Stage())(emotion="joy", action="backflip")
    assert not out["ok"] and out["did"] == ["emotion joy"]
    assert "Unknown action" in out["errors"][0]


def test_a_near_miss_feeling_is_shown_as_the_one_it_means() -> None:
    # Gemma reaches for "happy", "joyful", "curious", and reads "neutral joy" off the list as one name (2026-11-21).
    stage = Stage()
    avatar = avatar_tool(stage)
    assert avatar(emotion="happy") == {"ok": True, "did": ["emotion joy (for happy)"]}   # said back, so it learns the body's word
    assert stage.state["emotion"] == "joy"
    assert avatar(emotion="neutral joy")["did"] == ["emotion joy (for neutral joy)"]
    refused = avatar(emotion="loving")          # this body has no "love": still a refusal, with what it offers
    assert (refused["ok"], refused["offers"]["emotions"]) == (False, ["neutral", "joy"])


def test_none_means_no_move_not_a_move_to_refuse() -> None:
    # seen 2026-11-11: the small model fills the slot it does not want with "none"
    assert avatar_tool(Stage())(emotion="joy", action="none", look="None") == {"ok": True, "did": ["emotion joy"]}


def test_no_body_and_nothing_asked_are_said_plainly() -> None:
    assert "No avatar attached" in avatar_tool(Stage(on=False))(emotion="joy")["errors"][0]
    assert not avatar_tool(Stage())()["ok"]


def test_a_scenes_call_is_fitted_the_way_the_avatar_tool_would_take_it() -> None:
    for call in ({"emotion": "joy"}, {"action": "clap", "hold": 2.0}, {"action": "wink"}, {"look": "up left"},
                 {"emotion": "joy", "look": "center"}):
        assert fitted(call, OFFERS) == (call, []), call         # an action may be one of the body's reactions
    kept, left = fitted({"emotion": "wobbly", "action": "wave", "look": "sideways", "spin": "fast", "hold": 2}, OFFERS)
    assert kept == {"hold": 2}
    assert left == ["emotion wobbly (it shows: neutral, joy)", "action wave (it does: clap, wink)",
                    "look sideways (left, right, up, down or center)", "spin (a move is an emotion, an action or a look)"]
    assert fitted({"emotion": "joy", "action": "wave"}, OFFERS)[0] == {"emotion": "joy"}   # what it can do is kept
    for said in ("happy", "joyful", "neutral joy"):             # a near miss is shown as the feeling it means, not left out
        assert fitted({"emotion": said, "hold": 1}, OFFERS) == ({"emotion": "joy", "hold": 1}, []), said
    assert fitted({"emotion": "loving"}, OFFERS)[1] == ["emotion loving (it shows: neutral, joy)"]   # no love to mean
    # a move named bare in a scene arrives as an action; the body knows when it is a feeling or a look
    assert fitted({"action": "joy"}, OFFERS) == ({"emotion": "joy"}, [])
    assert fitted({"action": "left", "hold": 2}, OFFERS) == ({"look": "left", "hold": 2}, [])
    assert fitted({"action": "joy", "emotion": "neutral"}, OFFERS)[1] == ["action joy (it does: clap, wink)"]   # it has a feeling: not guessed


def test_the_models_script_is_staged_with_the_agent_to_play_after_its_words() -> None:
    staged = []
    scene = scene_tool(lambda script: staged.append(script) and [])
    answer = scene(script="look left | hold 3 | emotion joy | say Hi.")
    assert staged == ["look left | hold 3 | emotion joy | say Hi."]
    assert answer == {"ok": True, "note": "The scene plays now."}   # with ok, the agent ends the turn there: the scene is the reply
    left = scene_tool(lambda script: ["action wave (it does: clap, wink)"])(script="action wave | say Hi.")
    assert left["ok"] is True                    # it still plays, and the model reads what was left out for next time
    assert "without what this body cannot do: action wave (it does: clap, wink)" in left["note"]
    for part in ("say", "wait", "emotion", "action", "look", "hold", "on", "listen"):   # the model is shown every part of a beat
        assert part + " <" in scene.__doc__


def test_a_script_the_agent_cannot_read_goes_back_to_the_model_to_fix() -> None:
    def stage(script):
        raise ValueError("Beat 1: 'goodbye' is not in the line")

    assert scene_tool(stage)(script="say Hello. | on goodbye") == {"ok": False, "error": "Beat 1: 'goodbye' is not in the line"}
