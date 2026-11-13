"""Voice node: speaks through whichever Jaeger module fills the ``tts`` slot.

No loader lives here.  Today the slot is filled by jaeger-kokoro-tts (MLX on
Apple Silicon, CUDA or CPU elsewhere); replacing the module replaces the voice.
"""
from __future__ import annotations

from jaeger_agent.core.policy import KOKORO_LANG, KOKORO_VOICE, TTS_RATE


SHORT_CLIP_CHARS = 12


class MouthKokoro:
    def __init__(self) -> None:
        self.tts = None
        self.voice = KOKORO_VOICE
        self.info: dict[str, str] = {}      # module, version, model, compute
        self._opened = KOKORO_VOICE         # the voice the module was opened with: its own default
        self._short: dict[tuple, list] = {} # short clips by (voice, text), kept once made: "Oh," is instant the second time

    def load(self, say=print, *, voice: str = KOKORO_VOICE,
             language: str = KOKORO_LANG) -> None:
        from jaeger_agent.core.parts import open_part

        say("loading voice…")
        self.voice = self._opened = voice
        self._short = {}
        try:
            self.tts, info = open_part("tts", "open_voice", voice=voice, language=language)
        except SystemExit as exc:
            # A language-asset downloader calling sys.exit would end the
            # host's boot thread without reporting anything.
            raise RuntimeError("the voice module could not be initialized") from exc
        self.info = {**info, "model": voice}

    @property
    def sample_rate(self) -> int:
        """The voice's own rate; nothing downstream assumes one."""
        return getattr(self.tts, "sample_rate", TTS_RATE)

    def set_voice(self, voice: str) -> None:
        """Speak as ``voice`` from now on — another character has the floor.
        The module stays loaded; only the voice it is asked for changes."""
        self.voice = voice
        self.info = {**self.info, "model": voice}

    def synth(self, text: str, *, voice: str | None = None):
        """Yield float32 mono chunks at the module's sample rate, in the set
        voice or, for this line only, in ``voice``. A chunk comes through as
        the module made it, so its lip marks (``chunk.marks``) come with it."""
        voice = voice or self.voice
        key = (voice, text)
        if key in self._short:
            yield from self._short[key]
            return
        chunks = []
        # The module's own voice is asked for the old way, so a voice module
        # that takes no per-line voice still works until one is asked of it.
        options = {} if voice == self._opened else {"voice": voice}
        for chunk in self.tts.synth(text, **options):
            chunks.append(chunk)
            yield chunk
        # ponytail: 12 characters and 128 clips, as Jaeger Avatar's own voice
        # path keeps them; oldest out first. Raise both if longer openers recur.
        if len(text) <= SHORT_CLIP_CHARS:
            if len(self._short) >= 128:
                self._short.pop(next(iter(self._short)))
            self._short[key] = chunks


def build():
    return MouthKokoro()
