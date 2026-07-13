---
name: optimize-kernel-reliably
description: Plan, implement, benchmark, validate, and resume GPU or Triton kernel optimizations under Sparse-vLLM's reliable optimization framework. Use when Codex needs to select a kernel target, establish an oracle or benchmark harness, profile a kernel, generate and test optimization candidates, tune an existing correct kernel, run formal A/B qualification, integrate a winner, or audit or resume an optimization run. Support review, plan, execute, and resume modes.
---

# Optimize Kernels Reliably

Treat correctness, memory safety, support coverage, and measurement reliability as hard gates. Treat a well-supported no-improvement result as a valid outcome.

## Load the Contract

1. Read the repository `AGENTS.md` instructions.
2. Read [references/reliable-kernel-optimization-framework.md](references/reliable-kernel-optimization-framework.md) completely before planning or acting. Treat it as the canonical policy; do not duplicate or weaken it.
3. For `execute` or `resume`, read [references/gate-state-machine.md](references/gate-state-machine.md).
4. Before creating or changing a harness, manifest, run directory, or report, read [references/artifact-schema.md](references/artifact-schema.md).

## Select the Mode

Infer the narrowest mode authorized by the request:

- `review`: inspect only; do not edit or run GPU work.
- `plan`: create or revise the optimization plan and contracts; do not implement candidates.
- `execute`: establish evidence, implement candidates, run gates, and integrate only a qualified winner.
- `resume`: verify an existing run and continue from its last durable state.

An instruction such as "look first" or "do not change" always selects `review`. Do not expand `plan` into implementation without authorization.

## Establish the Run

For `execute`:

1. Inspect the target call chain, existing tests, benchmark entry points, production shapes, and repository state.
2. Freeze the target, supported input domain, hardware scope, objective, workload weights, thresholds, budgets, allowed paths, and commit mode.
3. Initialize a control run with `scripts/init_optimization_run.py`. Keep benchmark subruns beneath or referenced by that control run.
4. Persist every state transition with `scripts/update_optimization_state.py`; never use conversation memory as the sole progress record.
5. Preserve unrelated user changes. Stop if target files overlap unexplained worktree changes.

For `resume`, validate the control run before acting. Verify the framework hash, source hashes, state, budget, manifest exposure, and last completed gate. Start a new run if immutable evidence changed.

## Admit a Device

Before every GPU test, benchmark, or profiler task:

1. Run `scripts/check_gpu_idle.py` to inspect every visible device.
2. Select an idle device explicitly. Record the physical UUID and visible-device mapping.
3. If all devices are busy, wait only within the frozen budget. Report the situation instead of running on a busy device when the wait expires.
4. Repeat the snapshot before and after formal A/B. Reject a run whose utilization, processes, clocks, temperature, or power state violates its preregistered limits.

Do not silently select a different hardware class or relax admission thresholds.

## Build Evidence Before Candidates

Proceed in this order:

1. Freeze the semantic contract and supported input domain.
2. Build an independent semantic oracle and comparator contract.
3. Separate development, qualification, and integration replay manifests.
4. Establish a stable production baseline and immutable source identity.
5. Make the harness save raw, parsed, per-sample, aggregate, compile, command, and environment evidence separately.
6. Qualify the baseline before profiling or candidate work.
7. Profile the isolated kernel, pipeline, backend, and a small end-to-end workload.

If the baseline fails a required correctness or reliability gate, fix the baseline or harness as a distinct task. Do not tune around it.

## Generate and Test Candidates

Register each candidate before implementation with a stable ID, parent ID, bottleneck evidence, one primary change, predicted wins, predicted risks, correctness impact, falsification rule, and source hash.

Use the relevant specialist skill when available:

- Use `$write-triton-attention-kernel` for attention kernels.
- Use `$write-triton-gemm-kernel` for GEMM kernels.
- Use `$write-triton-softmax-kernel` for softmax kernels.
- Use `$write-triton-layernorm-kernel` for LayerNorm or RMSNorm kernels.
- Use `$optimize-triton-block-parameters` only after the kernel is correct.

Test one causal change at a time during screening. Register combinations as new candidates with explicit parents and rerun applicable gates from correctness onward. Never make a failed candidate silently dispatch to the baseline.

## Protect Qualification Integrity

Use the development manifest for adaptive screening. Lock code, thresholds, and a previously unexposed qualification manifest before formal A/B. Pair and interleave baseline and candidate measurements, keep raw rounds, and apply the preregistered confidence rule.

If a qualification result informs another code or parameter change, mark that manifest exposed. Do not reuse it as independent final evidence. Create a new run or fresh qualification manifest.

Validate completed artifacts with `scripts/validate_run_artifacts.py`. Use `scripts/compare_paired_variants.py` only when the harness emits matching `comparison_id`, `variant_id`, and raw positive latency samples.

## Integrate and Conclude

Integrate only a candidate that passes complete correctness, generalization, memory safety, formal performance, and backend or end-to-end gates. Dispatch by tensor geometry or explicit configuration, not model name.

Conclude with exactly one machine-readable decision:

- `selected`
- `no_improvement`
- `inconclusive`
- `blocked`

Report the winning or rejected candidates, qualification evidence, worst regressions, failures and reruns, residual limitations, source and manifest hashes, reproduction commands, and rollback point. Obey the frozen commit mode; do not commit, stash, or delete user work without authorization.
