# Gemma modular vs JaegerAgent 1.2.0

Run date: 2026-09-02 (US/Pacific)

> Historical full-agentic measurement. The latency and exit defects described
> here were subsequently addressed with a default-on tools toggle, exact-prefix
> startup warmup, and explicit MTMD/Llama shutdown. See
> `chatbot_mode_final.md` for the final three-run tool-free comparison and
> post-fix clean exit. The existing agentic route remains unchanged.

## Conclusion

JaegerAgent preserves the original Gemma modular pipeline's measured semantic
quality, speech recognition, and audio-output coverage while exercising the real
agentic runtime. It does **not** currently preserve the original latency or
process-stability benchmark.

## Controlled comparison

- Shared suite: `multimodal-agent-benchmark-v1` (19 cases).
- Same Gemma 4 E4B Q4_K_M weights, vision projector, Whisper large-v3-turbo
  fixtures, Kokoro PCM output, and Apple M1 Max host.
- The supplied VoiceLLM scorer independently recalculated correctness from the
  candidate's raw responses.
- JaegerAgent used its default `AgentRuntime`, runtime-owned memory, an 8,192-token
  context, 20 initially visible tools, and 105 available tools. Toolset scoping
  was enabled so the agent contract fits the context window.
- Baseline generation temperature was 0.7. JaegerAgent used its routing-safe
  runtime default of 0.0. Both used `max_tokens=160`.
- Baseline is the supplied single run. JaegerAgent was attempted three times;
  two runs completed all cases and one crashed during the memory case.

## Results

| Metric | Gemma modular baseline | JaegerAgent run 1 | JaegerAgent run 2 | Assessment |
|---|---:|---:|---:|---|
| Task accuracy | 19/19 | 19/19 | 19/19 | parity |
| Spoken-input WER | 2.90% | 2.90% | 2.90% | parity |
| Audio-output coverage | 100% | 100% | 100% | parity |
| Recorded case errors | 0 | 0 | 0 | parity within completed JSONs |
| Model load | 15.699 s | 22.647 s | 34.479 s | Jaeger mean 28.563 s, 1.82x slower |
| Median text first audio | 0.867 s | 9.458 s | 8.533 s | Jaeger mean 8.995 s, 10.38x slower |
| Memory first audio | 1.291 s | 19.767 s | 18.334 s | Jaeger mean 19.051 s, 14.75x slower |
| Median vision first audio | 1.420 s | 9.608 s | 8.992 s | Jaeger mean 9.300 s, 6.55x slower |
| Median spoken first audio | 1.843 s | 10.169 s | 10.216 s | Jaeger mean 10.193 s, 5.53x slower |

Accuracy is the suite's primary result. It is tied. Latency is therefore the
tie-breaker, and the original Gemma modular runner wins the current end-to-end
comparison.

## Agentic behavior observed

The memory case did not use framework-local chat history. Gemma emitted two real
`memory(remember, ...)` calls and later two `recall(...)` calls through
`AgentRuntime`; the response was correct. Other cases triggered real calculator
and web-search tools. This verifies that the measured candidate is the agentic
path rather than the marked non-agentic `llm-gemma` baseline node.

This also explains part of the latency gap: every request evaluates the
AgentRuntime system contract and scoped tool declarations, and some simple facts
incur tool round trips. The benchmark does not isolate prompt-prefill time from
generation, tool execution, or TTS.

## Stability finding

The completed JSONs each contain 19 cases with no recorded case error. However,
neither completed process exited cleanly: both aborted after writing the result
during llama.cpp Metal device teardown (`GGML_ASSERT([rsets->data count] == 0)`,
exit 134). The third attempt segfaulted during the memory case (exit 139) and did
not produce a result file.

Accordingly:

- Case-level reliability in completed results: 38/38, zero errors.
- Full-run completion: 2/3 attempts.
- Clean process exit: 0/3 attempts.

The post-write abort is not an inference error, but it is still a product/runtime
stability defect and must not be reported as “zero runtime errors” without that
qualification.

## Scope limits

As stated by the benchmark author, this suite measures semantic multimodal
performance after endpointing. It does not measure live AEC, barge-in behavior,
half-/quasi-full-/full-duplex overlap, streaming first-token/first-audio delivery,
or TTS naturalness. Those pipelines require the separate duplex simulation and
live hardware tests.

## Artifacts and commands

Candidate runner:

```bash
JAEGER_TOOLSET_SCOPING=1 python scripts/benchmark_multimodal_candidate.py \
  --benchmark-dir "/Users/jonathanjenkins/GITHUB/VoiceLLM-Playground/Multimodal LLM/benchmark" \
  --audio-dir "/Users/jonathanjenkins/GITHUB/jaeger_agent_omni/research/cases12" \
  --model-path "/Users/jonathanjenkins/.lmstudio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf" \
  --ctx 8192 --max-tokens 160 --temperature 0 --output candidate.json
```

Shared scorer/comparator:

```bash
.venv/bin/python "Multimodal LLM/benchmark/benchmark_multimodal.py" compare \
  "Multimodal LLM/benchmark/results/gemma_modular_full_20260902_150308.json" \
  candidate.json --output comparison.md
```
