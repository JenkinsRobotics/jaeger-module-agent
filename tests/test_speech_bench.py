"""The one speech benchmark: what voice and hearing run on, and how fast."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from jaeger_agent.bench import speech as bench

PARTS = {"voice": {"module": "v", "version": "1", "model": "a", "compute": "mlx"},
         "hearing": {"module": "h", "version": "2", "model": "b", "compute": "metal"}}


class StandInRuntime:
    """Speaks one second of silence in two chunks and 'hears' a fixed text."""

    tts = SimpleNamespace(tts=SimpleNamespace(sample_rate=1000))

    def __init__(self, hears):
        self.hears, self.heard_at = hears, []

    def status(self):
        return PARTS

    def synthesize_stream(self, text):
        yield np.zeros(400, dtype=np.float32)
        yield np.zeros(600, dtype=np.float32)

    def transcribe(self, audio, *, sample_rate):
        self.heard_at.append((len(audio), sample_rate))
        return self.hears


def test_measure_reports_the_parts_the_voice_timing_and_the_round_trip():
    runtime = StandInRuntime(hears=bench.TEXT)
    report = bench.measure(runtime, runs=2)

    assert report["parts"] == PARTS
    assert report["voice"]["audio_s"] == 1.0 and report["voice"]["chunks"] == 2
    assert 0 <= report["voice"]["first_audio_s"] <= report["voice"]["total_s"]
    assert report["voice"]["realtime_factor"] == round(report["voice"]["total_s"] / 1.0, 4)
    # Hearing is timed on the agent's own speech, at the voice's sample rate.
    assert runtime.heard_at == [(1000, 1000)] * 2
    assert report["hearing"]["clip_s"] == 1.0 and report["hearing"]["words_right"] == 1.0


def test_measure_shows_when_hearing_gets_the_words_wrong():
    report = bench.measure(StandInRuntime(hears="something else entirely"), runs=1)
    assert report["hearing"]["words_right"] < 0.2


def test_apple_silicon_must_run_voice_on_mlx_and_hearing_on_metal():
    assert bench.fast_path_problems(PARTS, apple_silicon=True) == []
    slow = {"voice": {**PARTS["voice"], "compute": "cpu"},
            "hearing": {**PARTS["hearing"], "compute": "cpu"}}
    problems = bench.fast_path_problems(slow, apple_silicon=True)
    assert len(problems) == 2 and "voice" in problems[0] and "mlx" in problems[0]
    assert "hearing" in problems[1] and "metal" in problems[1]
    assert bench.fast_path_problems(slow, apple_silicon=False) == []


def test_the_agent_command_runs_the_speech_benchmark(monkeypatch):
    from jaeger_agent.__main__ import main

    seen = []
    monkeypatch.setattr(bench, "main", lambda argv: seen.append(argv) or 0)
    assert main(["speechbench", "--runs", "1"]) == 0
    assert seen == [["--runs", "1"]]
