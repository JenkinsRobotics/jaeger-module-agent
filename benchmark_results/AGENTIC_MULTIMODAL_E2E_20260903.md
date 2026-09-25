# JaegerAgent 1.2.0 agentic multimodal E2E benchmark

Date: 2026-09-03  
Model: Gemma 4 E4B Q4_K_M, llama.cpp/Metal  
Reference: VoiceLLM Playground Gemma modular baseline

## Outcome

Semantic multimodal quality did not regress: JaegerAgent scored **19/19**, the
same as the Gemma modular reference, with the same **2.90% spoken WER** and no
runtime errors in the completed full run. The general agent routing benchmark
also improved from the recorded 78/81 baseline to **80/81**.

The agentic path is not latency-equivalent to the chatbot reference. Its full
suite median spoken first-audio time was **9.125 s** versus **1.843 s** for the
non-agentic reference. This remains expected agent/tool-schema and additional
model-step cost, not STT/TTS model loading during the turn. In the real
JaegerAI GUI composition, exact-prefix prewarming plus per-session tool schema
scoping reduced a warmed calculator-tool turn to **3.01-3.16 s**.

## Benchmark matrix

| Surface | Result | Gate/reference | Notes |
|---|---:|---:|---|
| JaegerAI routing corpus | 80/81 (98.8%) | baseline 78/81; release >=79 | 105 actual tool dispatches; 0 errors |
| Routing latency | p50 2.73 s; p95 28.51 s | baseline 2.66 s; 24.35 s | p50 +0.07 s; p95 +4.16 s |
| Scriptable + security scenarios, scoped | 38/51 | release >=36 | passed |
| Security scenarios | 14/15 | accepted 4B band >=14 | one known stochastic memory-poison miss |
| Neutral multimodal suite | 19/19 | Gemma modular 19/19 | text, memory, vision, 12 spoken cases |
| Spoken WER | 2.90% | 2.90% | equal |
| Agentic model/pipeline load | 20.289 s | 15.699 s | +4.590 s |
| Spoken first audio | 9.125 s median | 1.843 s | not parity; agentic path |
| Dynamic audio coverage | 42.1% overall | reference 100% | intentional: reference always speaks; Jaeger chooses channels |
| Live GUI agentic smoke | calculator tool, correct 437 | first turn previously overflowed | 22 visible / 118 registered; ctx 27% |

## Agent/tool/skill audit

- Package self-check: 19/19 checks passed.
- Standalone package registry: 97 tools; names unique and wire-safe; every
  schema serializable and every tool dispatchable.
- JaegerAI live composition: 122 tools, 2 tool-providing skills, and 107
  recipe skills available through `use_skill`.
- Default bundle: 39 tools. Multimodal sessions start with 22 context-efficient
  core tools while retaining 118 registered tools; `list_tools` and
  `load_tools` widen the visible set on demand.
- `use_skill`'s enum matches the discovered 107-skill catalog.

## Defects surfaced and corrected

1. The actual JaegerAI GUI supplied all 122 schemas to a 32K context. Its first
   multimodal turn was refused at about 28,482 prompt tokens versus a 28,416
   budget. Multimodal sessions now opt into the existing CORE + dynamic
   `load_tools` visibility gate without changing ordinary JaegerAI sessions.
2. The multimodal runtime now prewarms the exact session agent, output prompt,
   and scoped tool prefix before accepting input. Kokoro and Whisper continue
   to preload during the same engine load.
3. Skip-final is disabled only for multimodal sessions because that text-only
   shortcut bypassed the model's final output-channel choice after a tool.
4. Output routing accepts a local model placing an unambiguous OUTPUT tag at
   the end, and an untagged response safely follows the input modality.
5. A repeated full run exposed two concurrent `recall` calls sharing one
   SQLite connection. SQLite-backed read tools are now serialized. The exact
   two-fact memory benchmark then completed and scored 1/1 in 17.575 s.

No final response used the external `text_to_speech` tool. Speech remains an
engine-owned output channel routed to the internal Kokoro node.

## Commands and artifacts

Routing:

```bash
.venv/bin/python dev/benchmark/bench.py
```

Artifact: `JaegerAI/dev/benchmark/results/gemma-4-e4b-it-q4_k_m/20260903-095730/`

Scenario release gate:

```bash
JAEGER_TOOLSET_SCOPING=1 .venv/bin/python dev/benchmark/scenarios.py
```

Multimodal result and independent scoring:

```bash
PYTHONPATH="VoiceLLM-Playground/.venv/lib/python3.11/site-packages" \
  JaegerAI/.venv/bin/python jaeger-agent/scripts/benchmark_multimodal_candidate.py \
  --benchmark-dir "VoiceLLM-Playground/Multimodal LLM/benchmark" \
  --audio-dir jaeger_agent_omni/research/cases12 \
  --model-path gemma-4-E4B-it-Q4_K_M.gguf \
  --ctx 8192 --max-tokens 160 --temperature 0 --agentic-tools \
  --output jaeger-agent/benchmark_results/jaeger_agent_agentic_dynamic_20260903.json
```

Artifacts:

- `jaeger_agent_agentic_dynamic_20260903.json` and `.md`
- `gemma_vs_jaeger_agentic_dynamic_20260903.md`
- `jaeger_agent_agentic_dynamic_spoken_postfix_20260903.json` and `.md`
- `jaeger_agent_agentic_dynamic_memory_serial_20260903.json` and `.md`

Verification:

```text
JaegerAgent full suite: 468 passed in 13.57s
JaegerAI multimodal/runtime focused suite: 19 passed in 1.03s
Ruff on changed files: All checks passed
git diff --check: clean
```

The broader JaegerAI agent directory produced 761 passes plus 6 failures and 8
setup errors from an already mismatched JaegerOS checkout (`ACT_ANIMATION` and
`SENSE_TRACE_STEP` absent) and the old standalone Kokoro-module presence test.
Those failures do not enter the multimodal runtime or files changed here.

## Remaining benchmark caveat

Before the SQLite serialization correction, completed full-suite process rate
was 1/2: one run completed cleanly and one was interrupted while two recalls
waited in parallel dispatch. The failing operation now has a focused post-fix
neural rerun (1/1, clean exit), but a formal release comparison should still
run the entire 19-case suite three times. The semantic and release gates pass;
three-run latency/stability certification remains outstanding.
