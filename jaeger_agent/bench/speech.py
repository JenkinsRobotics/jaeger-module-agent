"""One benchmark for the agent's voice and hearing.

    python -m jaeger_agent speechbench [--runs N] [--json FILE]

Says which module, version, model and compute each part runs on, how fast the
voice is, and how fast and how well hearing takes back the agent's own speech.
No speaker and no microphone.  Exits 1 when a part is off the fast path.
"""
from __future__ import annotations

import argparse
import difflib
import json
import platform
import statistics
import sys
import time
from typing import Any

import numpy as np

TEXT = ("Octopuses have three hearts and blue blood. Two of the hearts pump blood "
        "through the gills, while the third sends it around the rest of the body.")

# ponytail: Apple Silicon is the only rule so far; add the CUDA rule when the
# agent is first run on that hardware.
FAST_ON_APPLE_SILICON = {"voice": "mlx", "hearing": "metal"}


def fast_path_problems(parts: dict[str, dict[str, str]],
                       apple_silicon: bool | None = None) -> list[str]:
    """Why a part is slower than this machine allows; empty when all is well."""
    if apple_silicon is None:
        apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
    if not apple_silicon:
        return []
    return [f"{part} runs on {parts[part]['compute']}, expected {want} on Apple Silicon"
            for part, want in FAST_ON_APPLE_SILICON.items()
            if parts[part]["compute"] != want]


def _words(text: str) -> list[str]:
    return "".join(c for c in text.lower() if c.isalnum() or c.isspace()).split()


def measure(runtime: Any, *, runs: int = 3) -> dict[str, Any]:
    """Time a loaded SpeechRuntime; the median of ``runs`` is reported."""
    rate = runtime.tts.tts.sample_rate
    voice, hearing, audio, heard = [], [], np.zeros(0, dtype=np.float32), ""
    for _ in range(runs):
        started, first, chunks = time.perf_counter(), None, []
        for chunk in runtime.synthesize_stream(TEXT):
            first = first or time.perf_counter() - started
            chunks.append(chunk)
        voice.append((first or 0.0, time.perf_counter() - started))
        audio = np.concatenate(chunks)
        started = time.perf_counter()
        heard = runtime.transcribe(audio, sample_rate=rate)
        hearing.append(time.perf_counter() - started)
    audio_s = len(audio) / rate
    total_s = round(statistics.median(t for _, t in voice), 4)
    return {
        "parts": runtime.status(),
        "voice": {"first_audio_s": round(statistics.median(f for f, _ in voice), 4),
                  "total_s": total_s, "audio_s": round(audio_s, 2), "chunks": len(chunks),
                  "realtime_factor": round(total_s / audio_s, 4)},
        "hearing": {"clip_s": round(audio_s, 2),
                    "decode_s": round(statistics.median(hearing), 4),
                    "words_right": round(difflib.SequenceMatcher(
                        None, _words(TEXT), _words(heard)).ratio(), 3),
                    "text": heard},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jaeger_agent speechbench", description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--json", metavar="FILE", help="also write the report here")
    args = parser.parse_args(argv)

    from jaeger_agent.core.speech import SpeechRuntime

    runtime = SpeechRuntime()
    started = time.perf_counter()
    runtime.load(lambda _message: None)
    load_s = round(time.perf_counter() - started, 2)
    measure(runtime, runs=1)                                # warm-up, dropped
    report = {"load_s": load_s, **measure(runtime, runs=args.runs)}
    report["problems"] = fast_path_problems(report["parts"])

    for part, info in report["parts"].items():
        print(f"{part:8} {info['module']} {info['version']}  model {info['model']}  "
              f"compute {info['compute']}")
    v, h = report["voice"], report["hearing"]
    print(f"load     {load_s} s")
    print(f"voice    {v['audio_s']} s of speech in {v['total_s']} s "
          f"(x{v['realtime_factor']} of real time), first audio {v['first_audio_s']} s")
    print(f"hearing  {h['clip_s']} s clip in {h['decode_s']} s, "
          f"{h['words_right']:.0%} of the words right")
    for problem in report["problems"]:
        print(f"NOT ON THE FAST PATH: {problem}")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=1)
    return 1 if report["problems"] else 0
