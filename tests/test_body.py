"""The agent in a body: the body is whatever fills the ``animation`` slot.

The agent carries no body code of its own. With a module in the slot it gets
a face, the avatar and scene tools and the body's own list of what it can do;
with nothing in the slot it is a voice with no face.

Written November 22, 2026 with the move from Jaeger Avatar, and not yet run.
"""
from __future__ import annotations

import base64
import queue
import time
from types import SimpleNamespace

import numpy as np

from jaeger_agent.body import Device
from jaeger_agent.body import loop as body_loop

OFFERS = {"emotions": ["neutral", "joy", "thinking", "listening"], "reactions": ["wink"], "actions": []}


def fill_slots(monkeypatch, **packages):
    """Stand-in for JaegerOS slot discovery: slot name -> the module filling it."""
    import jaeger_os.core.modules as modules

    monkeypatch.setattr(modules, "resolve_slot_module",
                        lambda slot, suffix="": packages.get(slot))


class Player:
    """Stands in for the animation module's player: keeps what it was asked."""

    def __init__(self) -> None:
        self.calls: list = []

    def play(self, samples, rate, marks=(), cues=()) -> None:
        self.calls.append(("play", len(samples), rate, list(marks), list(cues)))

    def hold(self) -> None:
        self.calls.append(("hold",))

    def release(self) -> None:
        self.calls.append(("release",))

    def stop(self) -> None:
        self.calls.append(("stop",))

    def wait(self, timeout=None) -> bool:
        return True

    def close(self) -> None:
        self.calls.append(("close",))


class Body:
    """Stands in for what the animation slot's open_body returns."""

    compute = "a test"
    name = "Mochi"
    character = {"who": "You are Mochi, an AI character. You are a small companion robot.", "voice": "af_heart"}
    activity = {"thinking": "thinking", "listening": "listening"}
    offers = OFFERS

    def __init__(self, bundle, **options) -> None:
        self.bundle, self.options, self.player, self.told, self.on_cue = bundle, options, Player(), [], None

    def control(self, op, value=None, ttl_ms=5000):
        self.told.append((op, value))
        return {"ok": True}

    def close(self) -> None:
        self.player.close()


class Host:
    """Stands in for host.AgentHost: keeps what the device sent, and hands it the events a test puts in."""

    def __init__(self) -> None:
        self.sent, self.q, self.ready, self.first, self.on_control = [], queue.Queue(), True, None, None

    def start(self, first) -> None:
        self.first = first

    def listen(self):
        return self.q

    def send(self, data) -> None:
        self.sent.append(data)

    def close(self) -> None:
        self.q.put(None)


def settle(done, seconds=2) -> None:
    end = time.time() + seconds
    while time.time() < end and not done():
        time.sleep(0.01)
    assert done()


def audio(text, samples=60, **more):
    pcm = (np.full(samples, 0.5) * 32767).astype("<i2").tobytes()
    return {"event": "audio", "text": text, "rate": 1000, "pcm": base64.b64encode(pcm).decode(),
            "marks": [], "cues": [], **more}


def animation(monkeypatch):
    fill_slots(monkeypatch, animation=SimpleNamespace(
        __name__="some_animation_module", __version__="9.9", open_body=Body))


def test_the_body_is_whatever_fills_the_animation_slot(monkeypatch) -> None:
    animation(monkeypatch)
    host = Host()
    device = Device(host, "a/bundle", checksum="abc", speaker="the speaker", delay_s=0.1)

    assert device.body.bundle == "a/bundle"
    assert device.body.options == {"checksum": "abc", "speaker": "the speaker", "delay_s": 0.1}
    assert device.part == {"module": "some_animation_module", "version": "9.9", "compute": "a test"}
    first = device.first
    assert (first["op"], first["session"], first["voice"]) == ("start", "device", "af_heart")
    assert "body" not in first and "open_mic" not in first        # a face, and the agent's own half duplex
    assert "a small companion robot" in first["persona"]          # who she is, as her bundle says
    assert "joy, thinking, listening" in first["persona"] and "wink" in first["persona"]
    assert "avatar tool" in first["persona"]

    device.start()
    try:
        assert host.on_control("emotion", "joy", 5000) == {"ok": True}   # the agent's avatar tool moves this body
        assert device.body.told == [("emotion", "joy")]
        device.body.on_cue({"emotion": "joy"})                     # a scene move's moment has come: the agent makes it
        assert host.sent == [{"op": "cue", "call": {"emotion": "joy"}}]
    finally:
        device.close()


def test_with_nothing_in_the_slot_the_agent_is_a_voice_with_no_face(monkeypatch) -> None:
    fill_slots(monkeypatch)
    host = Host()
    device = Device(host, "a/bundle", name="Nori", who="You are Nori, an AI character.")
    assert device.body is None and device.part == {}
    assert device.first["body"] is False                           # the worker offers no avatar or scene tool
    assert device.first["persona"].startswith("You are Nori, an AI character.")
    assert "avatar tool" not in device.first["persona"] and "body" not in device.first["persona"].lower()

    device.start()
    try:
        host.q.put(audio("Hi."))                                   # her voice: the agent's own speaker plays it
        host.q.put({"event": "assistant", "text": "Hi."})
        host.q.put({"event": "done"})
        device.text("Hello")
        device.stop()
        assert host.sent == [{"op": "text", "text": "Hello"}, {"op": "interrupt"}]
        assert device.turns.acquire(timeout=2)
    finally:
        device.close()


def test_an_older_module_in_the_slot_that_has_no_body_to_open_is_the_same(monkeypatch) -> None:
    fill_slots(monkeypatch, animation=SimpleNamespace(__name__="jaeger_animation", __version__="0.3.0"))
    assert Device(Host(), "a/bundle").body is None


def test_her_voice_goes_to_the_bodys_player_and_the_agent_hears_when_it_has_played(monkeypatch) -> None:
    animation(monkeypatch)
    host = Host()
    device = Device(host, "a/bundle")
    device.start()
    try:
        host.q.put({"event": "scene", "text": "begin"})
        host.q.put(audio("One.", cues=[{"at": 0, "call": {"emotion": "joy"}}]))
        host.q.put({"event": "voiced", "n": 3})                    # the agent waits, its microphone shut
        host.q.put({"event": "scene", "text": "end"})
        host.q.put({"event": "state", "text": "thinking"})
        settle(lambda: {"op": "played", "n": 3} in host.sent)
        settle(lambda: ("emotion", "thinking") in device.body.told)   # the feeling her pack gives for thinking
        assert device.body.player.calls[:4] == [
            ("hold",), ("play", 60, 1000, [], [{"at": 0, "call": {"emotion": "joy"}}]), ("release",), ("release",)]
        device.hear(np.zeros(160, dtype=np.float32))               # every frame goes to the agent: turn-taking is its own
        assert host.sent[-1]["op"] == "audio" and "pcm" in host.sent[-1]
        device.text("Hello")
        assert ("stop",) in device.body.player.calls               # typed over her: what she was saying stops
    finally:
        device.close()


def test_with_a_body_each_conversation_opens_with_an_avatar_call() -> None:
    made = body_loop.Loop(lambda line: None)
    assert made.examples and made.examples[0][2][0][0] == "avatar"   # main() empties this for a host with no body
