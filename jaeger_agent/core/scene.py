"""A scene: lines to speak with moves timed to them, performed in order.

A script is steps in order, split by ``|`` or by line breaks (they mean the
same):

    look left | hold 3
    look center | emotion joy | say So, what did you do today?
    say I spent mine learning to juggle. | action spin | on juggle

``say <line>`` speaks a line. ``wait <seconds>`` is quiet. Any other step
with a name and a value is a move for the body (the avatar tool takes
``emotion``, ``action`` and ``look``), handed as written to the body's tool.
A move rides on what comes next: it lands as the next line or quiet starts,
and a last move lands as the last line ends. ``hold <seconds>`` holds the
moves before it that long while the scene waits. ``on <word>`` puts the
moves before it on that word of the line just said.

The same scene may be given as a list of beats, each with its own line,
quiet and moves: ``[{"say": ..., "wait": ..., "look": ..., "on": ...}]``.

Each line is voiced whole and its moves ride on the sound as cues
(``clip.cues``: ``[{"at": seconds from the clip's start, "call": {...}}]``),
beside the lip marks. Whoever plays the clip makes the call when the clip's
own clock reaches ``at``, so move, lips and sound keep one clock.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterator

import numpy as np

MAX_WAIT_S = 30
TAIL_S = 0.2     # the quiet a scene's last move lands in, when nothing follows it
# What the conversation remembers a scene by, so a question asked in one can be answered.
DIRECTION = "(Stage direction: you performed a scene you were given. These were your lines.)"
_WORDS = ("say", "wait", "hold", "on", "emotion", "action", "look")
_JOINED = re.compile(r",\s*(?=(?:wait|hold|on|emotion|action|look)\s)", re.IGNORECASE)


class Clip(np.ndarray):
    """A scene line's sound, with its lip ``marks`` and its ``cues``."""


class Line(str):
    """One thing to voice: a line with the cues that ride on it, or (``wait``
    above zero, no words) a quiet stretch. A cue is the body's call, with
    ``on`` naming its word when it has one."""

    def __new__(cls, text: str, cues: list | None = None, wait: float = 0.0):
        self = super().__new__(cls, text)
        self.cues, self.wait = cues or [], wait
        return self


def _seconds(value: Any) -> float | None:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if not isinstance(value, bool) and 0 <= seconds <= MAX_WAIT_S else None


def _read(script: str) -> list[dict]:
    """A written script as beats: steps in order, a move riding on what follows."""
    rows = [row for row in script.replace("\\n", "\n").splitlines()   # a model may write its line breaks as the two characters \n
            if not row.strip().startswith("#")]
    beats: list[dict] = []
    moves: dict = {}                      # named, and waiting for something to ride on

    def beat(found: dict) -> None:
        nonlocal moves
        beats.append({**found, **({"cues": [moves]} if moves else {})})
        moves = {}

    number = 0
    steps = []
    for step in "\n".join(rows).replace("|", "\n").splitlines():
        # A model may name two moves in one step ("emotion joy, action bow"); a spoken line keeps its commas.
        steps += [step] if step.strip().lower().startswith("say ") else _JOINED.split(step)
    for step in steps:
        if not step.strip():
            continue
        number += 1
        key, _, value = step.strip().partition(" ")
        key, value = key.lower(), value.strip()

        def bad(why: str) -> ValueError:
            return ValueError(f"Step {number}: {why}")

        if not value:
            raise bad(f"'{key}' needs something after it" if key in _WORDS
                      else f"'{key}' needs a value, as in 'action {key}'")
        if key == "say":
            beat({"say": value})
        elif key in ("wait", "hold"):
            seconds = _seconds(value)
            if seconds is None:
                raise bad(f"{key} is 0 to {MAX_WAIT_S} seconds")
            if seconds:                   # "wait 0" passes no time: the moves ride on what follows it
                if key == "hold" and moves:
                    moves["hold"] = seconds
                beat({"wait": seconds})
        elif key == "on":
            if not moves:
                raise bad("'on' follows the move it places")
            if not beats or "say" not in beats[-1]:
                raise bad("'on' needs a line before it")
            if value.lower() not in beats[-1]["say"].lower():
                raise bad(f"'{value}' is not in the line")
            beats[-1].setdefault("cues", []).append({**moves, "on": value})
            moves = {}
        else:
            moves[key] = value
    if moves:
        beat({"wait": TAIL_S})
    return beats


def beats(script: Any) -> list[dict]:
    """The scene as checked beats, each ``{"say"?, "wait"?, "cues"?}``; a
    ValueError names the first step or beat that cannot be read."""
    if isinstance(script, str):
        found = _read(script)
    else:
        found = []
        for number, given in enumerate(script or [], 1):
            def bad(why: str) -> ValueError:
                return ValueError(f"Beat {number}: {why}")

            if not isinstance(given, dict):
                raise bad("a beat is a set of named parts")
            say, wait = given.get("say", ""), given.get("wait", 0)
            if not isinstance(say, str):
                raise bad("say is the line, as words")
            if _seconds(wait) is None:
                raise bad(f"wait is 0 to {MAX_WAIT_S} seconds")
            if not say.strip() and not wait:
                raise bad("needs a line to say or a time to wait")
            cues = list(given.get("cues") or [])
            moves = {k: v for k, v in given.items() if k not in ("say", "wait", "cues")}
            if set(moves) - {"on"}:       # the beat's own moves ride on its line
                cues.insert(0, moves)
            for cue in cues:
                if cue.get("on") and str(cue["on"]).lower() not in say.lower():
                    raise bad(f"'{cue['on']}' is not in the line")
            found.append({**({"say": say.strip()} if say.strip() else {}), **({"wait": wait} if wait else {}),
                          **({"cues": cues} if cues else {})})
    if not found:
        raise ValueError("A scene needs at least one beat")
    return found


def lines(script: Any, fit: Callable[[dict], dict] | None = None) -> list[Line]:
    """The scene as what is voiced, in order: each beat's line, then its quiet.
    ``fit`` is the body's say on each call a cue will make: it hands back the
    call as the body can make it (the same, or with parts left out; a call
    with no move left is dropped), or raises a ValueError to refuse the scene."""
    out = []
    for beat in beats(script):
        if "say" in beat:
            out.append(Line(beat["say"], beat.get("cues")))
        if beat.get("wait"):
            out.append(Line("", None if "say" in beat else beat.get("cues"), wait=beat["wait"]))
    for line in out if fit else []:
        fitted = [({**fit({k: v for k, v in cue.items() if k != "on"})}, cue.get("on")) for cue in line.cues]
        line.cues = [{**call, **({"on": on} if on else {})} for call, on in fitted if set(call) - {"hold"}]
    return out


def _cues(line: Line, seconds: float) -> list[dict]:
    # ponytail: a word's moment is the share of the letters before it; the
    # voice module's own word timings if a move is seen to land off its word.
    def at(cue: dict) -> float:
        return round(seconds * line.lower().index(str(cue["on"]).lower()) / len(line), 3) if cue.get("on") else 0

    timed = [{"at": at(cue), "call": {k: v for k, v in cue.items() if k != "on"}} for cue in line.cues]
    return sorted(timed, key=lambda cue: cue["at"])


def voiced(line: Line, synth: Callable[[str], Any], rate: int) -> Iterator[Clip]:
    """The line's sound as one clip with its marks and cues on it."""
    if line.wait:
        # ponytail: quiet goes out as silent sound, so every player keeps time
        # with no new message to learn; a 'pause' message if the bytes matter.
        sound, marks, seconds = np.zeros(int(line.wait * rate), dtype=np.float32), [], 0.0
    else:
        parts, marks, n = [], [], 0
        for part in synth(str(line)):
            marks += [[round(a + n / rate, 3), round(b + n / rate, 3), kind]
                      for a, b, kind in getattr(part, "marks", [])]
            parts.append(np.asarray(part, dtype=np.float32))
            n += len(part)
        sound = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        seconds = n / rate
    if not len(sound):           # the voice made nothing of it: no player can play an empty clip
        return
    clip = sound.view(Clip)
    clip.marks, clip.cues = marks, _cues(line, seconds)
    yield clip
