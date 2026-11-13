"""Real models, no stand-ins: the agent speaks, hears itself, and is on the fast path.

Runs wherever the voice and hearing modules are installed (jaeger-agent[multimodal]);
skipped on a text-only install.  No speaker and no microphone are used.
"""
from __future__ import annotations

import pytest
from jaeger_os.core.modules import resolve_slot_module

from jaeger_agent.bench import speech as bench
from jaeger_agent.core.config import MultimodalConfig
from jaeger_agent.core.speech import SpeechRuntime

_missing = [slot for slot, opener in (("tts", "open_voice"), ("stt", "open_hearing"))
            if not hasattr(resolve_slot_module(slot), opener)]
pytestmark = pytest.mark.skipif(
    bool(_missing), reason=f"no Jaeger module offers the {' / '.join(_missing)} part here")


def test_the_agent_speaks_and_hears_itself_on_the_fast_path():
    # base.en keeps this to seconds; compute is a property of the build, not the model.
    runtime = SpeechRuntime(MultimodalConfig(stt_model="base.en"))
    runtime.load(lambda _message: None)
    report = bench.measure(runtime, runs=1)

    assert set(report["parts"]["voice"]) == {"module", "version", "model", "compute"}
    assert report["hearing"]["words_right"] >= 0.9, report["hearing"]["text"]
    assert report["voice"]["realtime_factor"] < 1.0        # faster than real time
    assert bench.fast_path_problems(report["parts"]) == []
