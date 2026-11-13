"""Where the agent's voice goes.  The app decides; the engine only fans out.

An output is anything with ``play(chunk, rate)``, ``wait()`` and ``stop()``:
a speaker, a robot's sound card, an avatar's lip sync.  ``play`` gets mono
float32 at ``rate`` Hz and may block until the output is ready for the next
chunk, which keeps later outputs in the list in step with the first.
``wait`` returns once everything given has been heard; ``stop`` cuts it off.

Sound comes in the same way round: the app feeds ``engine.push_audio``.
"""
from __future__ import annotations


class Speaker:
    """A sound card on this machine; ``device=None`` is the system default."""

    def __init__(self, device: int | str | None = None) -> None:
        self.device = device
        self._busy = False

    def play(self, chunk, rate: int) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "sounddevice is required for playback; install jaeger-agent"
            ) from exc
        if self._busy:
            sd.wait()                  # chunk N finishes while N+1 was made
        sd.play(chunk, rate, device=self.device)
        self._busy = True

    def wait(self) -> None:
        if self._busy:
            import sounddevice as sd

            sd.wait()
            self._busy = False

    def stop(self) -> None:
        if self._busy:
            import sounddevice as sd

            sd.stop()
            self._busy = False


__all__ = ["Speaker"]
