# Kernel Optimization Artifact Contract

## Contents

1. Control run
2. Optimization specification
3. State and hypotheses
4. Manifest index
5. Per-sample results
6. Formal comparison data
7. Decision

## 1. Control run

Use one immutable control directory per optimization attempt:

```text
<run_dir>/
  optimization_spec.json
  optimization_state.json
  state_transitions.jsonl
  run_info.json
  case_manifest.json
  hypotheses.jsonl
  command_log.jsonl
  raw_outputs.jsonl
  parsed_outputs.jsonl
  per_sample_results.jsonl
  aggregate_metrics.json
  compile_metadata.jsonl
  decision.json
  report.md
  runs/
  ncu/
  nsys/
```

Use `schema_version="1.0"` in every JSON object and JSONL row. Write state and decision files atomically. Append immutable events to JSONL files. Do not replace raw evidence when parsing or reporting.

## 2. Optimization specification

Freeze these fields before candidate implementation:

```json
{
  "schema_version": "1.0",
  "mode": "execute",
  "target": {
    "name": "kernel name",
    "call_path": "public wrapper to kernel",
    "source_paths": ["repo/relative/path.py"],
    "supported_input_domain": "explicit shape, dtype, layout, mask, and error contract"
  },
  "hardware_scope": ["GPU model and compute capability"],
  "objective": {
    "primary_metric": "latency_ms",
    "direction": "minimize",
    "workload_weights": {"bucket": 1.0},
    "minimum_improvement": 0.03,
    "maximum_case_regression": 0.05,
    "confidence_level": 0.95
  },
  "budget": {
    "max_candidates": 8,
    "gpu_hours": 4.0,
    "wall_hours": 8.0,
    "max_retries_per_case": 1
  },
  "authorization": {
    "allowed_paths": ["repo/relative/path.py", "tests/path.py"],
    "commit_mode": "never"
  }
}
```

Use only `never`, `checkpoint`, or `final` for `commit_mode`. Record later scope changes by creating a new run; do not edit the frozen copy.

## 3. State and hypotheses

`optimization_state.json` must contain:

- `run_id`, `state`, `previous_state`, and `updated_at`;
- passed and failed gates with evidence paths;
- active candidate and candidate status counts;
- budget limits and current use;
- `next_action` and `stop_reason`.

Append every state change to `state_transitions.jsonl` before atomically replacing the current state file.

Each `hypotheses.jsonl` row must contain:

- stable `variant_id`, `parent_variant_id`, and `source_hash`;
- bottleneck evidence and one primary change;
- predicted win, predicted risk, and correctness impact;
- falsification rule;
- status and evidence paths.

Compute `source_hash` as SHA-256 of the compact, key-sorted JSON object mapping every target repository-relative source path to its file SHA-256. `run_info.json` keeps the initial mapping and `baseline_source_hash`; an active candidate row keeps the aggregate hash for its current source tree.

## 4. Manifest index

`case_manifest.json` indexes three independently hashed sets:

```json
{
  "schema_version": "1.0",
  "sets": {
    "development": {"path": "runs/dev/cases.json", "sha256": "...", "exposed": true},
    "qualification": {"path": "runs/formal/cases.json", "sha256": "...", "exposed": false},
    "integration": {"path": "runs/integration/cases.json", "sha256": "...", "exposed": false}
  }
}
```

Each case requires `case_id`, `required`, `expected_status`, input and layout fields, seed, and suite. Formal performance cases also require a stable `comparison_id` shared by baseline and candidates.

## 5. Per-sample results

Every `per_sample_results.jsonl` row requires:

```text
schema_version, case_id, required, variant_id, stage,
status, failure_kind, expected_status, gate_result,
attempt, first_attempt_ref, reason
```

Allowed top-level statuses are:

```text
success
invalid_input
model_failed
parse_failed
metric_failed
skipped_by_policy
```

Use `gate_result=pass` only when the observed status and all stage-specific checks match the preregistered expectation. A required negative-input case may pass with `status=invalid_input`; a required valid case normally requires `status=success`.

Use operator-specific `failure_kind` values such as `compile`, `ptxas`, `launch`, `oom`, `timeout`, `nonfinite`, `incorrect`, `oob`, `race`, `noisy_measurement`, `profiler_parse`, or `infrastructure`. Preserve the first failed attempt and link bounded reruns through `first_attempt_ref`.

## 6. Formal comparison data

For paired comparison, emit one raw row per case and variant with:

```text
schema_version, comparison_id, case_id, variant_id,
latency_samples_ms, pair_order, device_snapshot_ref, status
```

Samples must be positive finite paired blocks from the same timing boundary, with equal baseline and candidate sample counts. Baseline and candidate need identical comparison IDs. Keep the interleaving order so environment drift can be audited. Bootstrap paired samples within each fixed manifest case; do not resample cases and silently change the preregistered workload distribution.

Report per-case ratios, weighted or unweighted geometric mean as preregistered, worst regression, confidence interval, failed or missing pairs, and absolute latency. Do not aggregate failed or missing required pairs as if they were absent from the workload.

## 7. Decision

Create `decision.json` only when concluding. Require:

```text
schema_version, run_id, decision, selected_variant,
reason, gate_evidence, qualification_manifest_hash,
source_hash, rollback_point, limitations
```

Allow only `selected`, `no_improvement`, `inconclusive`, or `blocked`. `selected` requires an unexposed qualification manifest and complete integration evidence. `no_improvement` requires completion of the preregistered search; budget exhaustion is `inconclusive`.
