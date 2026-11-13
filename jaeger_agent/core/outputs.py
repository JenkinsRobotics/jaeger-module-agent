"""Typed final-output routing for multimodal JaegerAgent faces.

Final speech is not a tool action. The model returns one compact channel
directive with its final answer, and the engine routes the resulting text to
its own output nodes. A turn-scoped flag lets the normal tool-availability
gate hide external speech tools while this self-contained face is running.
"""

from __future__ import annotations

import contextvars
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Literal

DYNAMIC_OUTPUT_PROMPT = """

# Dynamic output routing
This face can emit text, speech, or both. You choose per turn:
- Each user turn begins with `[input: text]` or `[input: speech]`; use it as
  context when choosing the most natural response channel.
- For `[input: speech]`, normally answer with SPEECH so the user hears the
  reply. Choose BOTH if a written copy is useful or requested. Use TEXT alone
  only when the user explicitly requests no speech, or the answer is primarily
  code or a large table that should be read on screen.
- For `[input: text]`, normally answer with TEXT. If the user asks you to say,
  speak, read aloud, or answer by voice, use SPEECH or BOTH.
- Begin the final response with exactly one directive: `[OUTPUT:TEXT]`,
  `[OUTPUT:SPEECH]`, `[OUTPUT:BOTH]`, or `[OUTPUT:SILENT]`.
- Put the response after the directive. SILENT has no response body.
- Do not call `text_to_speech` for final user-facing speech. That external
  narration tool is intentionally unavailable in this multimodal face.
Choose naturally from the request, conversational context, and input modality.
The harness removes the directive and routes the content through its own output
nodes. These rules override the base text-only/default-TTS guidance.
"""

OutputMode = Literal["dynamic", "speech", "text", "mirror"]


@dataclass(frozen=True)
class OutputDecision:
    display_text: str
    speech_text: str
    channels: tuple[str, ...]
    source: str


_MULTIMODAL_OUTPUT_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "jaeger_agent_multimodal_output_active", default=False
)
_OUTPUT_DIRECTIVE = re.compile(
    r"^\s*\[OUTPUT:(TEXT|SPEECH|BOTH|SILENT)\]\s*", re.IGNORECASE
)
_TRAILING_OUTPUT_DIRECTIVE = re.compile(
    r"\s*\[OUTPUT:(TEXT|SPEECH|BOTH|SILENT)\]\s*$", re.IGNORECASE
)


@contextmanager
def multimodal_output_scope() -> Iterator[None]:
    """Mark schema building and execution as owned by a multimodal face."""
    token = _MULTIMODAL_OUTPUT_ACTIVE.set(True)
    try:
        yield
    finally:
        _MULTIMODAL_OUTPUT_ACTIVE.reset(token)


def multimodal_output_active() -> bool:
    """Whether final output is currently owned by the multimodal engine."""
    return _MULTIMODAL_OUTPUT_ACTIVE.get()


def split_output_directive(reply: str) -> tuple[str, str]:
    """Return ``(channel, body)``; tolerate a model putting its tag last."""
    clean = (reply or "").strip()
    match = _OUTPUT_DIRECTIVE.match(clean)
    if match is not None:
        return match.group(1).upper(), clean[match.end():].strip()
    # Tool-result finalization occasionally makes small local models append
    # the directive despite being told to prefix it. The choice is still
    # unambiguous, so route it and keep the control marker out of the UI/TTS.
    match = _TRAILING_OUTPUT_DIRECTIVE.search(clean)
    if match is not None:
        return match.group(1).upper(), clean[:match.start()].strip()
    return "", clean


def transform_output_content(reply: str, transform: Callable[[str], str]) -> str:
    """Restyle final content without allowing a filter to lose its channel."""
    channel, body = split_output_directive(reply)
    if not channel:
        return transform(body)
    marker = f"[OUTPUT:{channel}]"
    if channel == "SILENT":
        return marker
    transformed = (transform(body) or "").strip()
    return f"{marker} {transformed}".strip()


def _dynamic_decision(
    reply: str,
    input_modality: Literal["text", "speech"],
) -> OutputDecision:
    """Parse a channel directive, falling back to the user's channel."""
    clean = (reply or "").strip()
    if clean.upper() in {"[SILENT]", "<SILENT>", "[NO_RESPONSE]"}:
        return OutputDecision("", "", (), "model-silent")

    channel, content = split_output_directive(clean)
    if not channel:
        if input_modality == "speech":
            return OutputDecision(
                "", clean, ("speech",), "input-modality-fallback"
            )
        return OutputDecision(clean, "", ("text",), "model-text-fallback")

    channel = channel.lower()
    if channel == "silent":
        return OutputDecision("", "", (), "model-output-directive")
    if channel == "speech":
        return OutputDecision("", content, ("speech",), "model-output-directive")
    if channel == "both":
        return OutputDecision(
            content,
            content,
            ("text", "speech"),
            "model-output-directive",
        )
    return OutputDecision(content, "", ("text",), "model-output-directive")


# A tag the model made up for itself ("<emotion:joy>", "</b>"): short, plain
# and closed. Never a tool call — those carry "|", "{" or the name tool_call.
_PLAIN_TAG = re.compile(r"</?(?!tool_call\b)[A-Za-z][\w:\- ]{0,40}>")
_PLAIN_TAG_ARRIVING = re.compile(r"<(?:/?[A-Za-z][\w:\- ]{0,40})?/?")


def _speakable(text: str) -> str:
    """``text`` without made-up tags: a voice would read them out."""
    return " ".join(_PLAIN_TAG.sub(" ", text or "").split())


def decide_output(
    *,
    mode: OutputMode,
    input_modality: Literal["text", "speech"],
    reply: str,
) -> OutputDecision:
    """Resolve a typed model choice or one configured compatibility policy."""
    decision = _decide_output(mode=mode, input_modality=input_modality, reply=reply)
    spoken = _speakable(decision.speech_text)
    if spoken == " ".join(decision.speech_text.split()):
        return decision
    return OutputDecision(decision.display_text, spoken, decision.channels, decision.source)


def _decide_output(
    *,
    mode: OutputMode,
    input_modality: Literal["text", "speech"],
    reply: str,
) -> OutputDecision:
    clean = (reply or "").strip()
    if mode == "dynamic":
        return _dynamic_decision(clean, input_modality)
    # A previous turn may leave routing guidance in the model's history.
    # Control directives never belong in captions or synthesized audio,
    # including when the user switches to a fixed output mode.
    channel, clean = split_output_directive(clean)
    if channel == "SILENT":
        clean = ""
    if mode == "speech":
        return OutputDecision(clean, clean, ("text", "speech"), "configured-speech")
    if mode == "text":
        return OutputDecision(clean, "", ("text",), "configured-text")
    if mode == "mirror" and input_modality == "speech":
        return OutputDecision(clean, clean, ("text", "speech"), "mirrored-speech")
    return OutputDecision(clean, "", ("text",), "mirrored-text")


_ANY_OUTPUT_DIRECTIVE = re.compile(r"\[OUTPUT:(TEXT|SPEECH|BOTH|SILENT)\]", re.IGNORECASE)
_SENTENCE_END = re.compile(r"[.!?…]+[\"')\]]*\s+|\n+")
_CLAUSE_END = re.compile(r"[,;:—]\s+")
# Where a speaker may breathe in an opening that has no comma: before a joining
# word, once this many words are out.
# ponytail: six joining words, so "you | and I" can be cut wrongly; a phrase
# model if that is heard. Fewer words first = sooner sound but a shorter head
# start for the rest of the sentence.
_BREATH = re.compile(r"\s+(?=(?:and|but|because|when|while|which)\s)", re.IGNORECASE)
_BREATH_WORDS = 4
_NOT_FOR_THE_VOICE = re.compile(r"<|```")
# the last one closes Gemma's bare ``<call:name{…}>`` form
_TOOL_CALL_END = re.compile(r"<tool_call\|>|</tool_call>|<\|/tool_call\|>|\}\s*/?>")
_HELD_TAIL = 24  # enough of a muted stretch to catch an end mark or directive split across pieces


class StreamedSpeech:
    """Hands out finished sentences of a reply while the model still writes it.

    Feed it the text pieces as they arrive; each call returns the sentences
    that may be spoken now. :func:`decide_output` still rules on the finished
    reply — :meth:`rest` says what of it has not been spoken yet.
    """

    def __init__(self, *, mode: OutputMode, input_modality: str) -> None:
        self._model_chooses = mode == "dynamic"
        # what happens to text that carries no directive — same as decide_output
        self._on = mode == "speech" or (
            mode in ("dynamic", "mirror") and input_modality == "speech")
        self._held: bool | None = None  # what _on was before a tool call muted the voice
        self._buf = ""
        self._opened = False  # the reply's first words have gone to the voice
        self.spoken: list[str] = []

    def feed(self, piece: str) -> list[str]:
        self._buf += piece
        out: list[str] = []
        arriving = len(self._buf) + 1   # where a tag that is still arriving starts, if one is
        while True:
            # A directive is a hard break: one turn can hold several model
            # calls, and each call opens with its own.
            while (match := _ANY_OUTPUT_DIRECTIVE.search(self._buf)) is not None:
                out += self._sentences(self._buf[:match.start()], whole=True)
                if self._held is not None:
                    self._on, self._held = self._held, None
                if self._model_chooses:
                    self._on = match.group(1).upper() in ("SPEECH", "BOTH")
                self._buf = self._buf[match.end():]
            if self._held is not None:
                if (end := _TOOL_CALL_END.search(self._buf)) is None:
                    self._buf = self._buf[-_HELD_TAIL:]
                    self.spoken += out
                    return out
                self._on, self._held = self._held, None
                self._buf = self._buf[end.end():]
            # ponytail: a "<" or a code fence mutes the voice until the tool
            # call closes or the next directive — tool calls travel as text
            # here, and reading one aloud is worse than speaking that reply
            # late. A lone "<" ("3 < 5") mutes the rest; rest() still says it.
            if (junk := _NOT_FOR_THE_VOICE.search(self._buf)) is None:
                break
            # A tag the model made up is dropped and the voice keeps going:
            # muting the rest held a whole reply back until it was finished.
            if (tag := _PLAIN_TAG.match(self._buf, junk.start())) is not None:
                before, after = self._buf[:junk.start()], self._buf[tag.end():]
                gap = "" if not before or before[-1:].isspace() or after[:1].isspace() else " "
                self._buf = before + gap + after
                continue
            if _PLAIN_TAG_ARRIVING.fullmatch(self._buf, junk.start()):
                arriving = junk.start()   # it may still become one: wait for its end
                break
            out += self._sentences(self._buf[:junk.start()], whole=True)
            self._held, self._on = self._on, False
            self._buf = self._buf[junk.start() + 1:]
        # a directive that is still arriving stays in the buffer
        cut = self._buf.rfind("[")
        tail = self._buf[cut:] if cut >= 0 else ""
        if not (tail and "]" not in tail and len(tail) < 16
                and "[OUTPUT:".startswith(tail.upper()[:8])):
            cut = len(self._buf)
        cut = min(cut, arriving)
        out += self._sentences(self._buf[:cut], whole=False, keep=self._buf[cut:])
        self.spoken += out
        return out

    def _sentences(self, text: str, *, whole: bool, keep: str = "") -> list[str]:
        out: list[str] = []
        start = last_end = 0
        for end in _SENTENCE_END.finditer(text):
            words = len(text[last_end:end.end()].split())
            # "Sure.", "Dr." and "1." are not worth a breath of their own. An
            # opening "Oh!" is: it is voiced while the rest is still written
            # (as Jaeger Avatar's old path does).
            # ponytail: only "!", "?" and "…" mark an opener, so "Dr." and "1."
            # stay safe; a one-word "Sure." still waits for the next sentence.
            opener = (words == 1 and not out and self._on and not self._opened
                      and any(mark in end.group() for mark in "!?…"))
            if words >= 2 or opener:
                out.append(text[start:end.end()].strip())
                start = end.end()
            last_end = end.end()
        if whole:
            if text[start:].strip():
                out.append(text[start:].strip())
        else:
            # The opening goes out at its first clause when no sentence has
            # ended yet: a short first breath is written and voiced sooner
            # (as Jaeger Avatar's old path does; measured alone it did not
            # close the gap to that path — see the "First words" latency note).
            # A one-word clause ("Oh,") counts too: the voice node keeps short
            # clips once made, so it is heard at once the second time.
            # ponytail: any word that starts with a letter, so "1, 2, 3" stays
            # whole; a list of interjections if a lone "Well," sounds wrong.
            if not out and self._on and not self._opened:
                clause = _CLAUSE_END.search(text)
                if clause and (len(text[:clause.end()].split()) >= 2
                               or text.lstrip()[:1].isalpha()):
                    out.append(text[:clause.end()].strip())
                    start = clause.end()
                # No comma at all: a long first sentence went to the voice
                # whole, and its first sound came 0.4-0.8 s later than a short
                # opening's. It goes out at its first breath point instead.
                elif clause is None:
                    for breath in _BREATH.finditer(text):
                        if len(text[:breath.start()].split()) >= _BREATH_WORDS:
                            out.append(text[:breath.start()].strip())
                            start = breath.end()
                            break
            self._buf = text[start:] + keep
        if out and self._on:
            self._opened = True
        return out if self._on else []

    def rest(self, speech_text: str) -> str:
        """What of the finished reply's speech has not been spoken yet."""
        # The reply is the last model call's text; what was spoken may also
        # hold earlier calls' words. Line the two up by words: the longest
        # end of what was said that is also the start of the reply.
        final = _speakable(speech_text).split()
        said = " ".join(self.spoken).split()
        for same in range(min(len(said), len(final)), 0, -1):
            if said[-same:] == final[:same]:
                return " ".join(final[same:])
        return " ".join(final)


__all__ = [
    "DYNAMIC_OUTPUT_PROMPT",
    "OutputDecision",
    "OutputMode",
    "StreamedSpeech",
    "decide_output",
    "multimodal_output_active",
    "multimodal_output_scope",
    "split_output_directive",
    "transform_output_content",
]
