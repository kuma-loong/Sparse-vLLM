# Gate State Machine

Use this operational state machine for `execute` and `resume`. The canonical requirements remain in [reliable-kernel-optimization-framework.md](reliable-kernel-optimization-framework.md).

## State transitions

| State | Required evidence to enter | Allowed work | Exit evidence |
| --- | --- | --- | --- |
| `discovery` | Frozen initial run identity and authorization | Inspect call chains, tests, workloads, devices, and existing artifacts | Ranked target evidence and draft semantic contract |
| `contract_frozen` | Supported domain, hardware, objective, thresholds, budget, and commit mode | Build manifests, oracle, comparators, and harness | Deterministic manifests and artifact tests |
| `harness_ready` | Oracle independence and harness schema validated | Run baseline correctness and measurement-repeatability gates | Baseline source hash and complete baseline artifacts |
| `baseline_qualified` | Required baseline cases pass their expected statuses | Profile isolated, pipeline, backend, and end-to-end paths | Measured bottleneck attribution and hotspot decision |
| `profiled` | Hotspot gate passes and each candidate has bottleneck evidence | Register and implement development candidates | Candidate ledger with falsification rules and source hashes |
| `candidate_screening` | Candidate passes compile and correctness smoke | Run bounded development screening and full qualification prerequisites | Locked survivor set or supported no-improvement result |
| `qualification_locked` | Code, thresholds, qualification manifest, environment limits, and statistics are frozen | Run formal paired A/B only | Complete unexposed qualification evidence |
| `integration` | At least one candidate passes formal gates | Validate wrapper, backend, CUDA Graph, model output, quality, and end-to-end metrics | Integration evidence and rollback point |
| `concluded` | Decision is supported by complete evidence or a declared blocker | No further mutation of the run | `decision.json`, final report, artifact validation |

## Transition rules

- Move only to the next state. Create a new run when an immutable input changes.
- Record the previous state, new state, timestamp, evidence paths, current candidate, budget use, next action, and reason.
- Keep only one active candidate. Mark every other candidate `proposed`, `rejected`, `qualified`, or `abandoned`.
- Do not enter `candidate_screening` when the baseline is incorrect, noisy, or missing required samples.
- Do not enter `qualification_locked` until every survivor passes the complete correctness, generalization, and memory-safety prerequisites.
- Mark a qualification manifest exposed immediately after its result informs another candidate change.
- Enter `concluded` with `inconclusive` when a budget expires. Do not relabel it as `no_improvement` unless the preregistered search and qualification obligations completed.

## Resume audit

Before resuming:

1. Validate all JSON and JSONL files.
2. Recompute the framework, manifest, and target source hashes.
3. Check that the current branch and worktree do not invalidate the recorded candidate.
4. Confirm that the selected GPU still satisfies the hardware scope.
5. Confirm that elapsed budget, candidate count, and retry counts are within limits.
6. Continue only from the first incomplete action in `optimization_state.json`.

Never overwrite a completed subrun, first failure, exposed qualification result, or prior state-transition record.
