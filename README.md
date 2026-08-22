# Hermes Agent Optimization: DeepSeek V4 Flash + StateM

An update-safe execution harness for Hermes Agent, tuned for DeepSeek V4 Flash 0731 while remaining useful across local and hosted OpenAI-compatible models.

This project does not replace Hermes Agent or fork its full source tree. It layers a portable configuration, a fail-closed source overlay, StateM lifecycle support, provider-aware request handling, tool-call hardening, and an optional local vision adapter over a compatible Hermes installation.

The objective is practical agent performance: fewer malformed tool calls, fewer repeated actions, safer recovery after timeouts, better continuity across long tasks, and stronger evidence that the requested work was actually completed.

## At a glance

| Area | Stock Hermes Agent | This optimized harness |
| --- | --- | --- |
| Execution | General agent loop | Bounded prepare, execute, verify, repair, handoff lifecycle |
| Long tasks | Conversation history and normal session persistence | StateM checkpoints, compact handoffs, fresh-context continuation, independent audit |
| Tool failures | Provider and model dependent | Argument normalization, ID repair, bounded recovery, repeated-call protection |
| Timeouts | Retry or provider error handling | Verify-before-retry semantics for potentially completed mutations |
| Model support | Broad provider support | Broad support plus capability-safe Frontier policy and DSV4-specific tuning |
| Reasoning | Provider/model configuration | DSV4 max reasoning with provider vocabulary clamping for other models |
| Observability | Normal logs and status | Bounded phase transitions for provider wait, tools, compaction, interruption, and completion |
| Updates | Local modifications can drift or be overwritten | Digest-verified overlay that reapplies only when clean and otherwise fails closed |
| Vision | Provider or computer-use dependent | Optional local LFM2.5-VL auxiliary vision endpoint |

## Architecture

```text
DeepSeek V4 Flash 0731 or another chat model
                    |
                    v
        Provider-aware transport layer
                    |
                    v
      Frontier + StateM execution harness
                    |
                    v
      Native Hermes tools, skills, memory,
       checkpoints, verification, and UI
```

DeepSeek-specific behavior stays in the transport and model configuration layers. The execution protocol and safety controls remain provider-neutral, so OpenRouter models, hosted APIs, local vLLM servers, LM Studio, Ollama, and other OpenAI-compatible endpoints can benefit without receiving unsupported DeepSeek fields.

Embedding and reranking endpoints are excluded from the agent protocol by default.

## Key features

### Frontier execution protocol

Complex work is guided through a bounded lifecycle:

1. Prepare the workspace and recover relevant state.
2. Localize the requirement, failure, or target component.
3. Form a compact plan with explicit success evidence.
4. Execute only the next useful action.
5. Verify the result against files, commands, or acceptance criteria.
6. Repair using a changed strategy when evidence fails.
7. Write a compact handoff before a context or session boundary.
8. Finish with a concise result, evidence, and any genuine blocker.

This protocol does not add a second hidden agent loop or require extra model calls. It contributes bounded instructions and phase state to the normal Hermes request path.

### StateM durable execution

StateM adds durable state to failure-prone and long-running work:

- Atomic private checkpoint writes
- Workspace-aware run isolation
- Compact phase packets instead of replaying an entire history
- Evidence receipts for completed actions
- Explicit pending clarification and authorization states
- Fresh-user-turn gating before protected work resumes
- Durable handoff across sessions and context compaction
- Repair state that records why the previous strategy failed
- Independent audit before progress is treated as verified

The model can resume from a small, machine-readable contract instead of reconstructing the task from an increasingly noisy transcript.

### Verify-before-retry tool effects

Tools that can change files, processes, services, or remote systems are treated as non-atomic effects. If a timeout or interruption happens after dispatch, the operation is recorded as `unknown` until Hermes can reconcile the result.

The harness uses stable operation identities and evidence checks to avoid blindly repeating a mutation that may already have succeeded. This is particularly important for shell commands, service restarts, file writes, and remote operations.

### Tool-call normalization and repair

The shared tool path handles common differences between OpenAI-compatible providers:

- Missing or null `tool_calls` containers
- Tuple and list tool-call representations
- Incrementally streamed names and JSON arguments
- Nested or aliased argument containers
- Duplicate or missing tool-call IDs
- Blank, malformed, or hallucinated tool names
- Empty calls paired with `finish_reason=tool_calls`
- Mixed batches containing both valid and invalid calls
- Incomplete JSON arguments
- Provider metadata that must be retained or stripped on replay

Recovery is bounded. Invalid output is not allowed to create an unlimited repair loop.

### Loop and stall protection

The harness tracks useful progress rather than counting commands alone:

- Cross-turn identical-call detection
- Repeated failure signatures
- Read-only action streaks
- Idempotent no-progress actions
- Path deviation from the stated task
- Maximum unproductive-action budgets
- Bounded repair attempts
- Final-response budget retained when a guardrail stops execution

When a strategy repeatedly fails, the model receives a concise recovery boundary and must change approach or hand off honestly.

### DeepSeek V4 Flash 0731 profile

The primary profile retains the behavior required by the custom DSV4 endpoint:

- `reasoning_effort: max`
- Preservation of structured `reasoning_content`
- DeepSeek-compatible replay of assistant tool turns
- Long provider-wait handling without premature cancellation
- Context-aware Frontier reasoning budgets
- Completion-token accounting for speculative or buffered output
- No changes to the separate DeepSeek/vLLM serving recipe

The harness improves execution quality and reliability. It does not modify model weights or claim to transform DSV4 into a different underlying model.

### Cross-model capability safety

The Frontier protocol applies to chat-capable models by default. Provider-specific request fields remain gated by the existing Hermes transport logic:

- Reasoning effort is clamped to each wire vocabulary.
- Gemini thinking configuration is emitted only for Gemini models.
- Gemini thought signatures are removed before replay to strict non-Gemini providers.
- Unsupported reasoning fields are not assumed for plain local endpoints.
- DeepSeek reasoning metadata remains separate from OpenRouter reasoning details.
- Empty tool-call arrays are removed for strict providers that reject them.

This lets weaker or less agent-specialized models benefit from the execution discipline without forcing one provider's schema onto another.

### Bounded reasoning disclosure

The desktop integration can show concise lifecycle information such as provider wait, tool execution, verification, repair, compaction, interruption, and terminal completion.

It does not expose private chain-of-thought. The UI surfaces phase, action, evidence, and blocker summaries that are useful for diagnosing a stalled session.

### Optional local vision

The repository includes an OpenAI-compatible adapter for `LiquidAI/LFM2.5-VL-3B-MLX-8bit`. It can provide a small local auxiliary vision path without sending images to OpenRouter or forcing the agent to open Chrome for ordinary image inspection.

Model weights are intentionally not included.

## Update safety

The customization bundle is designed to survive Hermes updates without restoring stale code wholesale.

- `overlay.patch` contains only the owned source differences.
- `manifest.json` records the expected SHA-256 overlay digest and preserved capabilities.
- `reapply.py` checks whether an update is active before touching the checkout.
- A reverse patch check detects when the overlay is already present.
- A forward patch check must succeed before any source is changed.
- Configuration invariants and external StateM/vision assets are audited separately.
- Conflicts create `NEEDS_ATTENTION` and leave upstream source untouched.
- No reset, checkout, stash restore, or broad configuration overwrite is performed.

The result is fail-closed update handling: upstream changes are preserved, and ambiguous customization drift requires reconciliation instead of silently reviving stale patches.

## Recorded benchmark results

The following baseline was recorded on 2026-08-21 using `deepseek-v4-flash-0731` with reasoning set to `max`.

### Long-horizon diagnostic

| Metric | Result |
| --- | ---: |
| Final verified requirements | **24 / 24** |
| Verified completion rate | **100%** |
| Fresh-context rounds | **3** |
| API calls | **146** |
| Provider errors | **0** |
| Runner timeouts | **0** |
| Guardrail events | **0** |

The task required implementation across three fresh-context rounds, durable handoffs, and an independent grader. Only auditor-passed requirements counted as progress.

This is a custom harness diagnostic. It is not a standardized Terminal-Bench or Long-Horizon Terminal-Bench score and should not be presented as one.

### Served-model perplexity diagnostic

| Metric | Result |
| --- | ---: |
| Chat-conditioned perplexity | **5.0319** |
| Scored WikiText-2 content tokens | **22,458** |
| Repeat-run perplexity | **5.0263** |

The score uses vLLM prompt log-probabilities with the fixed chat-template prefix and suffix subtracted. It is useful for regression comparison of this serving recipe, but it is not directly interchangeable with untemplated base-model perplexity.

### Buffered single-stream timing diagnostic

| Metric | Result |
| --- | ---: |
| Median time to first token | **20.51 s** |
| Median effective throughput | **37.44 tokens/s** |
| Mean effective throughput | **37.93 tokens/s** |
| Measured completions | **3 x 768 tokens** |

The endpoint buffered the streamed response, so a valid steady-state decode rate was not available. These figures describe end-to-end effective throughput and must not be represented as raw vLLM decode speed. Timing comparisons are invalid when other workers share the serving endpoint.

## What the scores do and do not mean

The results support the claim that the harness can complete the included long-horizon task with durable state, verification, and no recorded provider or guardrail failures in the baseline run.

They do not prove parity with a frontier closed model, universal performance across repositories, or a standardized terminal-agent benchmark result. Model quality, serving configuration, context length, endpoint load, quantization, and task design remain important variables.

Perplexity, execution success, and throughput are deliberately reported separately rather than merged into a synthetic intelligence score.

## Repository contents

- `config/config.example.yaml`: redacted multi-provider Hermes configuration template
- `customizations/frontier-harness/overlay.patch`: Frontier, StateM, tool, and lifecycle source overlay
- `customizations/frontier-harness/reapply.py`: fail-closed update reapplication controller
- `customizations/frontier-harness/manifest.json`: capability, provenance, and integrity manifest
- `bin/hermes-statem`: portable StateM launcher
- `bin/lfm25_vl_server.py`: local OpenAI-compatible LFM vision adapter
- `bin/start-lfm25-vl3b-local.sh`: local MLX vision service launcher
- `.env.example`: empty environment-variable placeholders

## Security and portability

This repository intentionally excludes:

- API keys and authentication files
- Private endpoint addresses
- SSH hosts and network topology
- Usernames and machine-specific workspace paths
- Chat history and session databases
- Logs, checkpoints, and StateM run data
- Model weights and local caches
- Launch-agent files
- The installed Hermes source checkout

All credentials and private endpoint values must be supplied locally through environment variables. Never commit the real `~/.hermes/config.yaml`, `.env`, `auth.json`, session data, or model cache.

## Installation outline

1. Install the compatible upstream Hermes Agent version.
2. Copy `.env.example` to an untracked `.env` and provide local values.
3. Adapt `config/config.example.yaml` without committing private endpoint information.
4. Place the customization bundle under the local Hermes customization directory.
5. Apply the overlay only when its clean-check succeeds.
6. Install StateM and the optional LFM vision dependencies separately.
7. Restart Hermes and verify the selected model, normal tools, checkpointing, and provider path.

Review `NEEDS_ATTENTION` after every Hermes update. Do not force an overlay across an upstream conflict.
