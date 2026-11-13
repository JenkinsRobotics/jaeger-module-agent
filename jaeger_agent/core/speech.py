"""Process-owned speech runtime for JaegerAgent's multimodal pipeline.

Applications host this object; they do not substitute their own STT or TTS
implementations into the conversational path.  The standalone ``listen`` and
``text_to_speech`` tools remain separate capabilities and are intentionally
not used here.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from .config import MultimodalConfig


class SpeechRuntime:
    """Own and prewarm the one Whisper/Kokoro pair used by agent turns."""

    def __init__(self, config: MultimodalConfig | None = None) -> None:
        self.config = config or MultimodalConfig()
        self._load_lock = threading.Lock()
        self._tts_lock = threading.Lock()
        self._playback_lock = threading.Lock()
        self._loaded = False
        self.stt: Any = None
        self.tts: Any = None

    def load(self, say: Callable[[str], None] = print) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            # Resolve the portable node folders through the engine's canonical
            # loader.  Import remains lazy so ``import jaeger_agent`` never
            # imports optional Whisper/Kokoro dependencies.
            from .engine import stt_mod, tts_mod

            self.tts = tts_mod.build()
            self.stt = stt_mod.build()
            self.tts.load(
                say,
                voice=self.config.kokoro_voice,
                language=self.config.kokoro_language,
            )
            self.stt.load(say, model_name=self.config.stt_model)
            self._loaded = True

    def transcribe(self, audio: Any, *, sample_rate: int = 16000) -> str:
        """Decode mono device PCM, resampling in the agent rather than the app."""
        if not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 192000:
            raise ValueError("sample_rate must be an integer between 8000 and 192000")
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size > sample_rate * 120 or not np.isfinite(samples).all():
            raise ValueError("STT input must be finite mono PCM, at most 120 seconds")
        if sample_rate != 16000:
            from scipy.signal import resample_poly

            divisor = math.gcd(sample_rate, 16000)
            samples = resample_poly(samples, 16000 // divisor, sample_rate // divisor)
        self.load()
        return str(self.stt.transcribe(samples)).strip()

    def transcribe_segments(self, audio: Any, *, language: str = "en",
                            initial_prompt: str | None = None,
                            abort_callback: Any = None) -> list[dict[str, Any]]:
        """Timestamped segments for a host's incremental STT.

        ``t0``/``t1`` are centiseconds.  The node's lock is the authoritative
        decode lock for all faces, including previews and whole utterances.
        """
        self.load()
        with self.stt._lock:
            segments = self.stt.model.transcribe(
                np.asarray(audio, dtype=np.float32).reshape(-1),
                language=language, prompt=initial_prompt, abort=abort_callback)
            return [
                {"text": str(s.text), "t0": s.start_ms // 10, "t1": s.end_ms // 10}
                for s in segments
            ]

    def status(self) -> dict[str, dict[str, str]]:
        """Which module, version, model and compute each part runs on."""
        self.load()
        return {"voice": dict(self.tts.info), "hearing": dict(self.stt.info)}

    def synthesize_stream(self, text: str) -> Iterator[np.ndarray]:
        """Yield each Kokoro chunk immediately, without buffering the reply."""
        self.load()
        with self._tts_lock:
            for chunk in self.tts.synth(text):
                yield np.asarray(chunk, dtype=np.float32).reshape(-1)

    def synthesize(self, text: str) -> np.ndarray:
        chunks = list(self.synthesize_stream(text))
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def speak(self, text: str, *, cancel_event: threading.Event | None = None) -> bool:
        """Play final speech through the same nodes, with bounded cancellation.

        OutputStream avoids sounddevice's process-global play/stop functions:
        cancelling this reply cannot stop another owner's unrelated audio.
        The host controls capture; this runtime owns synthesis and playback.
        """
        from .cancellation import current_cancellation
        from .policy import TTS_RATE

        turn_event = current_cancellation()
        def is_cancelled() -> bool:
            return bool((cancel_event is not None and cancel_event.is_set())
                        or (turn_event is not None and turn_event.is_set()))

        if not text or is_cancelled():
            return False
        while not self._playback_lock.acquire(timeout=0.05):
            if is_cancelled():
                return False
        try:
            if is_cancelled():
                return False
            import sounddevice as sd

            played = False
            stream = None
            rate = getattr(self.tts, "sample_rate", TTS_RATE)
            try:
                for chunk in self.synthesize_stream(text):
                    if is_cancelled():
                        return False
                    if not len(chunk):
                        continue
                    if stream is None:
                        stream = sd.OutputStream(
                            samplerate=rate, channels=1, dtype="float32",
                            latency="low", device=self.config.output_device)
                        stream.start()
                    # At most 20 ms of PCM per write, so device buffering does
                    # not turn a long sentence into an uninterruptible wait.
                    block = max(1, rate // 50)
                    for offset in range(0, len(chunk), block):
                        if is_cancelled():
                            return False
                        stream.write(chunk[offset:offset + block])
                        played = True
                return played
            finally:
                if stream is not None:
                    try:
                        stream.abort() if is_cancelled() else stream.stop()
                    finally:
                        stream.close()
        finally:
            self._playback_lock.release()


__all__ = ["SpeechRuntime"]
