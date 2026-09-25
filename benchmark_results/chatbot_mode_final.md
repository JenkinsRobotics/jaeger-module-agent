# JaegerAgent 1.2 chatbot-mode benchmark and lifecycle report

Date: 2026-09-02 (US/Pacific)

## Decision

The existing agentic pipeline remains the default and was not simplified.
`tools_enabled=false` selects a separate chatbot configuration of the same
JaegerAgent runtime: the model receives an explicit empty tool catalogue while
AgentRuntime continues to own the per-session transcript. JaegerAI exposes the
same choice as a default-on **Agentic tools** checkbox fixed for each
Stop/Start session.

The earlier latency was prompt/route cost, not audio transport cost. The
agentic candidate submitted about 15,842 characters of tool schemas (20 visible
tools) on elementary turns and sometimes performed additional model/tool
rounds. The tool-free lane removes those schemas and rounds without bypassing
the AgentRuntime memory boundary.

## Three-run result

All three no-tools candidate results scored 19/19, 2.90% spoken WER, 100% audio
coverage, and zero case errors with the shared scorer.

| Metric | Gemma modular baseline | Jaeger chatbot run 1 | run 2 | run 3 | Jaeger mean |
|---|---:|---:|---:|---:|---:|
| Model load | 15,698.95 ms | 14,184.50 ms | 11,610.53 ms | 11,850.19 ms | 12,548.41 ms |
| Text first audio | 866.64 ms | 884.12 ms | 939.79 ms | 945.50 ms | 923.14 ms |
| Memory first audio | 1,291.19 ms | 1,254.56 ms | 1,320.77 ms | 1,305.30 ms | 1,293.54 ms |
| Vision first audio | 1,420.47 ms | 1,303.25 ms | 1,308.30 ms | 1,250.56 ms | 1,287.37 ms |
| Spoken first audio | 1,842.93 ms | 1,736.90 ms | 1,779.11 ms | 1,791.03 ms | 1,769.01 ms |

The three-run mean loads 20.1% faster than the reference and reaches spoken
audio 73.9 ms (4.0%) sooner. Text is 56.5 ms slower on average, memory is
effectively equal (+2.4 ms), and vision is 133.1 ms faster. Run 3 is the final
post-fix run and returned `BENCHMARK_EXIT=0` after `ggml_metal_free:
deallocating`; there was no interpreter-teardown assertion or abort.

## Code decisions and locations

- `jaeger_agent/core/config.py:120`: default-on `tools_enabled` runtime setting.
- `jaeger_agent/core/runtime.py:283`: explicit `[]` only in chatbot mode; `None`
  preserves the full registered tool surface and current agentic behavior.
- `jaeger_agent/core/runtime.py:300` and
  `jaeger_agent/adapters/local_llama.py:634`: exact-prefix, one-token local
  Gemma warmup without a synthetic conversation turn.
- `jaeger_agent/core/engine.py:300`: startup owns component initialization; Gemma,
  Kokoro, Whisper, vision, and VAD are ready before `load()` returns. Duplicate
  STT/TTS primes were removed because their nodes already warm themselves.
- `jaeger_agent/nodes/vision-mmproj/runtime.py:34` and
  `jaeger_agent/core/engine.py:928`: close the child MTMD projector context before
  the parent Llama model. This fixes the previous ggml-metal exit-134 abort.
- `jaeger_agent/core/runner.py:116`: CLI switch
  `--agentic-tools/--no-agentic-tools`.
- `JaegerAI/jaeger_ai/core/mind_runtime.py:105`: isolated no-tools session map
  sharing the already-loaded model and LLM lock, without changing the existing
  `run_for_voice` agentic route.
- `JaegerAI/jaeger_ai/interfaces/pyside6/multimodal/window.py:278`: visible,
  default-on **Agentic tools** session toggle.
- `JaegerAI/jaeger_ai/interfaces/pyside6/multimodal/worker.py:108`: detach the
  closed projector from JaegerAI's shared Llama object without closing the
  chassis-owned runtime.

## Verification outputs

Final candidate command (same model, context, token cap, temperature, WAVs,
vision fixtures, Whisper, and Kokoro as the reference comparison):

```bash
PYTHONPATH="/Users/jonathanjenkins/GITHUB/VoiceLLM-Playground/.venv/lib/python3.11/site-packages" \
  /Users/jonathanjenkins/GITHUB/JaegerAI/.venv/bin/python \
  scripts/benchmark_multimodal_candidate.py \
  --benchmark-dir "/Users/jonathanjenkins/GITHUB/VoiceLLM-Playground/Multimodal LLM/benchmark" \
  --audio-dir "/Users/jonathanjenkins/GITHUB/jaeger_agent_omni/research/cases12" \
  --model-path "/Users/jonathanjenkins/.lmstudio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf" \
  --ctx 4096 --max-tokens 160 --temperature 0.7 \
  --no-agentic-tools \
  --output benchmark_results/jaeger_agent_chatbot_run3.json
```

Observed: `Cases: 19; errors: 0`, `BENCHMARK_EXIT=0`.

Other checks:

- jaeger-agent full suite with multimodal dependency path: `449 passed in
  13.84s`.
- jaeger-agent canonical selftest: `python tests/test_multimodal_selftest.py`
  → `23/23 checks passed`.
- JaegerAI focused runtime/bridge/window tests: `30 passed in 1.07s`.
- JaegerAI multimodal canonical selftest: `10/10 checks passed`.
- `python -c "import jaeger_agent"`: zero output, exit 0.
- grep for the legacy dotted multimodal package path across both repositories:
  zero matches.
- `git diff --check` in both repositories: clean.

JaegerAI's unrelated repository-wide suite currently stops during collection
on two files that predate this change: missing `jaeger_ai.core.models.lazy_deps`
and missing `jaeger_ai/agent/skills/computer_use_v1/computer_use.py`. The
multimodal focused suite above is green; those unrelated missing modules were
not repaired as part of this change.

## Artifacts

- `jaeger_agent_chatbot_run3.json`: raw final candidate
- `jaeger_agent_chatbot_run3.md`: independently scored final candidate
- `gemma_modular_vs_jaeger_chatbot_run3.md`: shared baseline comparison
- `jaeger_agent_chatbot_run3.log`: final process log including explicit Metal
  deallocation
