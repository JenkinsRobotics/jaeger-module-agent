"""Hearing node: hears through whichever Jaeger module fills the ``stt`` slot.

No loader lives here.  Today the slot is filled by jaeger-whisper-stt (Metal
on Apple Silicon, CUDA or CPU elsewhere); replacing the module replaces it.
"""
from __future__ import annotations

import numpy as np

from jaeger_agent.core.parts import HEARING_LOCK, open_part
from jaeger_agent.core.policy import STT_MODEL


class SttWhisper:
    def __init__(self) -> None:
        self.model = None                   # the module's hearing object
        self._lock = HEARING_LOCK           # every pass in the process, one at a time
        self.info: dict[str, str] = {}      # module, version, model, compute

    def load(self, say=print, *, model_name: str = STT_MODEL) -> None:
        say(f"loading hearing {model_name}…")
        self.model, info = open_part("stt", "open_hearing", shared=True,
                                     model=model_name)
        self.info = {**info, "model": model_name}

    def transcribe(self, audio, abort=None) -> str:
        """Words in ``audio``. ``abort()`` turning true stops a pass nobody needs."""
        with self._lock:
            if abort is not None and abort():
                return ""
            return " ".join(s.text for s in self.model.transcribe(
                np.asarray(audio, dtype=np.float32).reshape(-1), language="en",
                abort=abort)).strip()


def build():
    return SttWhisper()
