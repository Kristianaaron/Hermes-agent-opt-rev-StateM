# Hermes Agent Frontier Harness

An update-safe optimization layer for Hermes Agent. It keeps Hermes' native model, tool, skill, memory, desktop, and session features, then adds a bounded execution protocol, durable StateM handoffs, provider-safe request shaping, adaptive GLM reasoning, local GLiNER routing, and recovery guardrails.

The goal is not to make every prompt think harder. The goal is to make simple prompts fast, make tool-using work decisive, and make complex or risky work deliberate and verifiable.

This repository is a customization bundle rather than a fork of the full Hermes source tree. It is designed to be applied to a compatible Hermes checkout and safely reconciled after upstream updates.

## What is native and what is added

The following remain native Hermes capabilities:

- Provider and model selection
- Chat sessions, desktop UI, TUI, skills, plugins, and normal tool registration
- Workspace/project context
- Memory and todo tools
- Native streaming, checkpoints, and session persistence
- Native model-specific transports where supported by the upstream version

The following are non-native additions or behavior changes in this harness:

| Addition | What it improves |
| --- | --- |
| Frontier execution phases | Keeps long tasks oriented around the next useful action and explicit evidence instead of unbounded narration. |
| StateM lifecycle and handoffs | Resumes long work from compact durable state rather than replaying a huge, noisy transcript. |
| Verify-before-retry | Prevents a timeout after a mutation from causing an unsafe duplicate write or duplicate service action. |
| Tool argument normalization | Repairs common provider/model serialization differences before they become tool failures. |
| Repetition and no-progress guards | Stops repeated reads, status polling, stale-memory calls, and identical failed actions from consuming the turn. |
| Adaptive GLM effort routing | Uses low effort for clearly bounded work and high effort when complexity, risk, failure, or an explicit high request warrants it. |
| GLiNER semantic evidence | Adds a small local classifier signal for routing without sending a second model request. |
| Fast-turn context compaction | Reduces prefill and improves reusable KV/cache prefixes for short System-1 turns. |
| Provider wire normalization | Avoids strict-provider 400s caused by empty tool arrays and unsupported request fields. |
| Empty/partial-response recovery | Preserves an actionable failure explanation and completion state instead of returning a misleading blank result. |
| Update-safe overlay | Reapplies only known changes after an upstream update and fails closed on conflicts or drift. |
| Lifecycle observability | Shows bounded provider, tool, verification, repair, compaction, interruption, and completion state without exposing private chain of thought. |
| Optional local vision adapter | Adds a local OpenAI-compatible LFM2.5-VL path for image inspection without making vision mandatory. |

The harness does not change model weights, create a hidden second agent, or promise that a smaller model has frontier-model capabilities. It improves orchestration, request construction, context selection, and recovery around the selected model.

## Design in one view

```text
User turn
   |
   v
Turn classification and context selection
   |                         \
   |                          +--> GLiNER evidence (local, optional)
   v
Adaptive provider request
   |
   v
Frontier execution governor
   |
   +--> prepare -> inspect/localize -> act -> verify -> repair -> handoff
   |                                                     |
   +--> StateM checkpoint, receipt, clarification, or final result
```

The fast path is intentionally short. A simple prose answer or bounded local action can use compact context, a small tool set, and low reasoning. A complex implementation, failure recovery, risky mutation, or explicit high request can use the full context and high reasoning lane.

## Non-native feature details

### 1. Frontier execution protocol

The harness adds a bounded execution governor around the normal Hermes turn loop. It represents work as a sequence of evidence-producing phases:

1. **Prepare** — identify the workspace, recover only relevant durable state, and establish the task boundary.
2. **Inspect/localize** — inspect the smallest relevant set of files, processes, or resources.
3. **Act** — perform the next useful mutation or tool action.
4. **Verify** — check the result against files, command output, process state, or acceptance criteria.
5. **Repair** — if verification fails, change strategy rather than repeating the same action.
6. **Handoff** — write a compact progress packet when context, session, or operator attention is needed.

This improves completion quality by making “I ran a command” different from “the requested state was verified.” It also gives the model a bounded recovery budget and preserves a final-response budget when execution must stop.

### 2. Durable StateM execution

StateM is used as a durable execution layer for long-running or interruption-prone work. The harness integrates it with the frontier phases rather than treating it as a second chat history.

Added behavior includes:

- Atomic private checkpoint writes
- Workspace-aware run isolation
- Machine-readable phase packets
- Evidence receipts for completed actions
- Pending clarification and authorization states
- Real-user-turn gating before protected work resumes
- Compact handoffs across sessions and context compaction
- Repair records explaining why a previous strategy failed
- Independent audit before progress is treated as verified

The improvement is lower resume cost and less state contamination. A fresh turn can receive the current goal, known facts, pending action, last evidence, and next safe action without replaying tens of thousands of tokens of old tool traffic.

### 3. Verify-before-retry and idempotent effects

Tool effects are not assumed to be atomic. A process launch, file write, service restart, deployment, or remote mutation may complete even when the client sees a timeout or disconnect.

The harness therefore:

- Assigns stable operation identities and idempotency keys
- Records an effect as `unknown` when dispatch may have happened but confirmation is missing
- Runs reconciliation and verification before replaying an unknown effect
- Avoids blindly retrying a mutation that may already have succeeded
- Preserves partial or failed completion metadata for the next turn

This reduces duplicate writes, duplicate server starts, repeated deployments, and “retry until it works” behavior that can make a partially successful task worse.

### 4. Tool argument normalization and repair

The provider boundary and tool executor normalize common OpenAI-compatible representation differences, including:

- Tuple/list tool-call representations
- Incrementally streamed tool names and JSON arguments
- Nested or aliased argument containers
- Duplicate or missing tool-call IDs
- Blank, malformed, or hallucinated tool names
- Empty calls paired with `finish_reason=tool_calls`
- Mixed valid/invalid batches
- Incomplete JSON arguments
- Provider metadata that must be retained or removed during replay

Repair is bounded. A malformed output can be corrected once or classified as a failure; it cannot create an unlimited repair loop.

### 5. Repetition, stall, and path-deviation protection

The harness measures progress, not just the number of tool calls. It detects:

- Identical calls repeated across turns
- Repeated failure signatures
- Read-only loops such as `read_file` with no resulting change
- Idempotent no-progress actions
- Stale memory or patch-anchor retries
- Unchanged process polling
- Work that has drifted away from the requested path
- Mute/status/ping loops that never produce a useful user-facing update

The response is a concise recovery boundary, a changed strategy, or a truthful handoff. Work tools are allowed a different budget from housekeeping tools, and terminal output retains room for a final explanation.

### 6. Adaptive GLM Codex-style reasoning

`plugins/glm-codex-effort` adds model-aware `llm_request` middleware for GLM 5.3 Flash deployments whose chat template supports only Spark-legal `low` and `high` reasoning values.

The router exposes these internal lanes:

| Lane | Typical use | Context/tools | GLM effort |
| --- | --- | --- | --- |
| `fast` | Clearly bounded System-1 prose or a small deterministic action | Compact current-turn context and bounded tools | Thinking off |
| `standard_compact` | Recovery, overflow, or a compact tool continuation | Compact context and bounded tools | Low |
| `standard` | Normal agent work | Normal harness context and tools | Low |
| `full` | Complex planning, architecture, risky work, repeated failure, or explicit high | Full context and tool set | High |
| `finalize` | Late safety net: a bounded task is still running long after it escalated | No tools, explicit final-answer instruction | Thinking off (low on retry after an empty reply) |

Action budgets apply only to bounded (System-1) tasks. A bounded task that exceeds `fast_max_calls` gets compact low reasoning. One that exceeds `fast_hard_max_calls` escalates to the full harness instead of being cut off mid-change. `finalize` applies only after `fast_finalize_after_calls`. Normal harness work is never budget-clamped: Hermes' own `max_turns` and loop guardrails bound it.

The router also provides:

- A manual `high` override that remains sticky for the active task, including on entry points that do not supply a turn id
- Correction away from unnecessary low-lane escalation without demoting active high reasoning
- Natural-language continuation inheritance
- Preservation of the active lane across Hermes-generated “continue now” system rows
- Compact reasoning retained after an initial tool miss
- `clear_thinking: true` on GLM tool hops so prior reasoning is not needlessly re-deliberated
- Invalid GLM effort values clamped to supported values rather than falling through to an unintended maximum
- Provider capability gating so other models do not receive GLM-only fields
- A first tool failure in normal work keeps the full harness; only bounded tasks recover in the compact lane
- Repeated-failure escalation that still works if the overlay's harness internals are unavailable
- Tools already called in the turn stay declared in the bounded tool set, and a prose-only request keeps tools once the turn has tool history
- A fast-lane output cap (16384 by default) large enough for a whole `write_file`/`patch` payload
- Normal and high lanes keep the last few completed exchanges (`history_recent_exchanges`, default 3) when compacting long sessions

“Thinking off” here means the bounded fast/finalization request does not ask GLM for hidden reasoning. It does not disable the agent’s tool protocol, verification, or ordinary user-facing answer.

The router is semantic and state-aware; it is not a list of special-case keywords. Prompt length, requested outcome, actionability, failure state, risk, tool requirements, explicit effort, and prior active lane all contribute to the decision. The system-generated continuation fix is especially important: a UI “continue” event should resume the active complex lane instead of silently demoting the task to a fast or final lane.

### 7. GLiNER semantic routing evidence

`plugins/gliner-extract` supplies a pre-warmed local GLiNER classifier for lightweight semantic evidence. It classifies a request into categories such as:

- `quick_response`
- `bounded_operation`
- `complex`
- `risk`
- `ambiguous`

The result is evidence for the GLM router, not a replacement for the LLM. It runs locally, uses request-local inputs, and falls back to the normal harness path if the optional dependency or model is unavailable.

Hermes passes the same original request to every `llm_request` middleware and keeps the last returned payload. The GLiNER middleware therefore never returns a request, and the router pulls evidence directly through `classify()`. Routing works whichever plugin loads first. Cached text returns without touching the worker. A request never queues behind a prediction that is still running. A failed model load backs off for five minutes instead of retrying on every request.

Attached URL/file context is excluded from the routing text while remaining available to the model. This prevents a large pasted document, URL, or image description from being mistaken for a complex user intent merely because it increases input length.

The trade-off is a one-time startup/prewarm cost. After prewarm, the per-request classification is intended to be much cheaper than an additional LLM routing call. GLiNER is therefore used for fast semantic evidence, while the selected model remains responsible for the actual answer and tool plan.

### 8. Prefill, context, KV, and cache behavior

Fast turns use a self-contained, stable prefix and compact current-turn context. When continuity is required, the harness can retain conversational endpoints and relevant evidence without replaying every intermediate trace.

This improves:

- Prefill latency on short System-1 prompts
- Prefix stability for provider-side KV or response caching
- Token cost and input-token variance across tool hops
- Recovery from a large or contaminated prior session

The harness does not promise cache hits: cache eligibility and keying remain provider-specific. It improves the chance of reuse by avoiding unnecessary prefix churn and by keeping fast-turn requests self-contained.

### 9. Provider wire safety and empty-tool handling

Some strict OpenAI-compatible providers reject `tools: []`, while others accept or ignore it. The harness normalizes the final request at the wire boundary and after SDK request transformation:

- Omit genuinely empty top-level tool arrays
- Omit empty tool arrays nested in provider request bodies
- Preserve a valid placeholder when another request layer carries non-empty tools
- Avoid unsupported GLM or reasoning fields on providers that do not advertise them
- Preserve valid tool payloads and tool-only turns

This fixes a class of HTTP 400 errors such as: “`tools` must not be an empty array.” It does not fix authentication, quota, upstream outage, malformed credentials, or a provider that rejects a different field.

The SDK transform bypass also avoids an expensive typed walk for large provider payloads while preserving the intended wire format. This can reduce model-call overhead on very large conversations, but it is guarded by regression tests and final request normalization.

### 10. Empty, partial, and reasoning-only response recovery

The response path distinguishes a real empty provider response from a completed tool turn, a reasoning-only response, an interrupted turn, and a provider failure. It uses bounded retry/cost rules and retains visible failure or partial-completion metadata.

The improvement is diagnosability: a provider failure should not look like a successful blank answer, and a tool turn should not be mistaken for a finished user response. Provider authentication errors still need provider configuration to be corrected.

### 11. Lifecycle observability and UI behavior

The overlay adds bounded lifecycle state for:

- Provider wait and recovery
- Tool execution
- Verification and repair
- Context compaction
- Interruption and stale-turn fencing
- Completion and partial completion

Desktop-facing changes prefer concise status and evidence summaries. They do not expose private chain of thought. The update-safe bundle also includes the reasoning disclosure preference and provider-wait test fixes needed to keep the UI truthful about whether generation is active, waiting, recovering, or complete.

### 12. Optional local LFM2.5-VL vision adapter

The `bin/` scripts provide an optional OpenAI-compatible local adapter for `LiquidAI/LFM2.5-VL-3B-MLX-8bit`. It can support ordinary image inspection without routing every image to a hosted provider or forcing a browser workflow.

The adapter is auxiliary. It is not required for the frontier harness, and model weights are not included in this repository.

### 13. Update-safe source overlay

The customization is shipped as an explicit overlay instead of a silent fork:

- `overlay.patch` contains owned source differences
- `manifest.json` records provenance, capability claims, and SHA-256 digests
- `reapply.py` detects active updates before touching the checkout
- Reverse checks detect an already-applied overlay
- Forward checks must pass before source changes are made
- Three-way application fails closed on conflicts
- Configuration invariants and external runtime assets are audited separately
- No wholesale reset, checkout, stash restore, or broad configuration overwrite is performed

When upstream changes overlap with a customization, the correct result is `NEEDS_ATTENTION`, not silently reapplying stale code. This keeps upstream fixes and the local overlay reviewable.

## Installation outline

1. Install a compatible upstream Hermes Agent checkout. The current bundle records the reconciled upstream version in `customizations/frontier-harness/manifest.json`.
2. Copy `.env.example` to an untracked `.env` only if local environment values are needed.
3. Adapt `config/config.example.yaml` locally without committing private endpoints, API keys, or session data.
4. Place `customizations/frontier-harness` in the local Hermes customization directory.
5. Install the tracked plugins under `~/.hermes/plugins` or use the repository's plugin installation workflow:

   ```text
   plugins/glm-codex-effort
   plugins/gliner-extract
   ```

6. Install the optional GLiNER dependencies from `plugins/gliner-extract/requirements.txt` if semantic routing is desired, and list `gliner-extract` under `plugins.enabled` (see `config/config.example.yaml`). Without it the router still works, using harness evidence only.
7. Install StateM and optional LFM vision dependencies separately when those capabilities are desired.
8. Run the verifier before and after applying the overlay. Resolve `NEEDS_ATTENTION` rather than forcing the patch across an upstream conflict.
9. Set finite stream timeouts for the GLM provider (`stale_timeout_seconds`, `timeout_seconds`, `hard_timeout_seconds`; see the example config). An explicit `stale_timeout_seconds` overrides Hermes' built-in floors, so a very large value makes a dead stream look like a hung agent. `reapply.py` requires `hard_timeout_seconds` in the live config.
10. Restart Hermes and verify the selected model, normal tools, provider path, checkpoints, and GLM routing behavior.

The bundle is update-safe, not update-automatic: after each Hermes update, run the verifier and review any changed source or configuration before reapplying.

## Verification

The plugin tests are offline and do not make model calls:

```bash
python -m pytest -q \
  plugins/glm-codex-effort/test_glm_codex_effort.py \
  plugins/gliner-extract/test_gliner_extract.py
```

The installed checkout verifier is run against the local Hermes source tree, not this documentation repository:

```bash
python customizations/frontier-harness/verify_overlay.py \
  --repo "$HERMES_HOME/hermes-agent"
```

The expected overlay, verifier, GLM router, and GLiNER digests are recorded in the manifest. The current bundle was regression-tested with focused and broader adaptive-harness suites before publication.

## Repository contents

- `customizations/frontier-harness/overlay.patch` — owned Hermes source overlay
- `customizations/frontier-harness/reapply.py` — transactional, fail-closed update controller
- `customizations/frontier-harness/verify_overlay.py` — offline marker/import/digest verifier
- `customizations/frontier-harness/gateway_start.py` — compatibility-gated launcher
- `customizations/frontier-harness/manifest.json` — provenance, capability, and integrity manifest
- `plugins/glm-codex-effort/` — adaptive GLM lanes, continuation inheritance, and offline tests
- `plugins/gliner-extract/` — optional local semantic routing evidence and offline tests
- `config/config.example.yaml` — redacted configuration template
- `bin/hermes-statem` — portable StateM launcher
- `bin/lfm25_vl_server.py` — optional local vision adapter
- `bin/start-lfm25-vl3b-local.sh` — optional local MLX vision launcher
- `.env.example` — empty environment-variable placeholders

No OMP integration or OMP dependency is included in this harness. No session data, logs, private configuration, credentials, model weights, launch-agent files, or installed Hermes checkout is published.

## Security and portability

Never commit:

- API keys, OAuth tokens, authentication files, or private provider URLs
- Real `~/.hermes/config.yaml` or `.env` files
- Session databases, chat history, logs, checkpoints, or StateM run data
- Usernames, machine-specific paths, SSH hosts, or network topology
- Model weights or local caches
- Launch-agent files

The public bundle uses portable placeholders such as `$HERMES_HOME`, `$HOME`, and `$HF_HOME` for external assets. The local installation supplies actual paths at runtime.

## Scope and limitations

- GLiNER is optional evidence, not a guarantee that every prompt is classified correctly.
- Adaptive routing reduces avoidable reasoning and context cost; it cannot remove model latency caused by a slow provider or overloaded endpoint.
- A fast lane does not bypass tool safety, verification, authorization, or provider error handling.
- High reasoning is intentionally sticky for complex or explicitly high work until the task reaches a safe boundary.
- Provider authentication, quota, outage, and model availability errors still require provider-side fixes.
- Benchmark numbers, when present in project history, are diagnostic measurements for specific hardware, models, and workloads; they are not standardized benchmark claims.

## License and upstream relationship

This repository is a customization bundle for Hermes Agent. Review the upstream Hermes Agent license and the licenses of optional dependencies before redistribution. Upstream source remains the source of truth for the base agent; this repository documents and carries only the overlay and auxiliary integration artifacts.
