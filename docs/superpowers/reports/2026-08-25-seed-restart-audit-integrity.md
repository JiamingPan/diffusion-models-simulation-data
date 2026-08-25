# Seed-restart audit integrity fix

## Incident

Both stage-1 DiT-L16 seed-restart tasks completed training from 300k to 340k,
saved their target checkpoints, and validated the target checkpoint state. They
then exited with an incomplete-audit error because the final JSON lacked
`ema_restore`.

The audit had three independent writers in one process:

1. the EMA factory wrote `ema_restore`;
2. the accelerator load hook wrote the base resume contract;
3. the first-backward hook wrote `first_resumed_loss`.

Each writer replaced the entire JSON file from its own private dictionary. The
last writer therefore erased fields produced by the earlier writers. Atomic
rename prevented torn JSON, but it could not prevent this last-writer-wins
semantic loss.

## Writer contract

Shared report enrichment now uses one explicit contract:

- only rank 0 may write; `RANK`, `SLURM_PROCID`, and `LOCAL_RANK` are checked;
- an adjacent `fcntl.flock` covers the complete read-modify-write transaction;
- updates are a **shallow top-level merge**;
- repeating an identical key/value is idempotent;
- changing an existing top-level value is a loud error;
- publication uses a unique same-directory temporary file, file `fsync`,
  `os.replace`, and directory `fsync`.

The base contract, `ema_restore`, `first_resumed_loss`, and
`target_checkpoint_state` are now independent merge updates. Tests exercise
both real writer orders, concurrent processes, collisions, nonzero ranks, and
an interruption immediately before publication.

## Durable-storage decision

The incremental audit remains on the durable project filesystem rather than
node-local storage. A node-local-only audit would disappear when a GPU job dies,
which would remove the exact evidence needed to diagnose or recover a failed
run. The rank-0 contract, advisory lock, and same-directory durable replacement
protect the shared path. The Slurm job still uses one task, so the rank guard is
also a fail-closed assertion of the intended topology.

## Existing 340k recovery

`scripts/backfill_seed_restart_ema_audit.py` reconstructs only the missing
`ema_restore` object. Before writing, it verifies:

- a finite `first_resumed_loss` exists;
- a nonempty `target_checkpoint_state` exists;
- the recorded target checkpoint directory exists;
- the source checkpoint's EMA sigma profiles and burn-in match the audit;
- every exact source EMA snapshot exists at the recorded step.

It then shallow-merges `ema_restore` plus an explicit `audit_recovery` marker.
It never edits either the 300k source checkpoint or the validated 340k target.
This lets a retry recognize stage 1 as already complete instead of repeating
four GPU-hours per model.

## Repository-wide JSON-writer sweep

The sweep searched Python and Slurm entry points for `json.dump`,
`json.dumps(...write_text)`, audit/report path variables, and terminal-report
calls.

| Path class | Ownership found | Action |
| --- | --- | --- |
| Seed-restart `resume_audits/*.json` | Multiple field writers | Replaced whole-file writes with locked, rank-0, collision-detecting merges. |
| Terminal lifecycle reports | One producer job, but separate start/update/finalize processes | Added the same rank guard and a lock around every lifecycle read-modify-write; existing producer-ID and terminal-state checks remain. |
| Seed-restart completion JSON | One unique path per Slurm job and array task, written once after the audit passes | No merge needed; documented as single-writer output. |
| Config manifests, notebook files, evaluation metrics | Whole artifact produced by one invocation | No shared incremental writer was found. These retain a single-invocation ownership assumption. Concurrent invocations targeting the same output directory remain an operational risk and should use distinct output roots. |

The earlier terminal-status work did not catch this incident because the resume
audit did not use the terminal-report lifecycle at all, and its tests checked
terminal state transitions rather than ownership of independent top-level
fields. This change adds a general shared-field merge primitive and exercises
the actual seed-restart writer ordering.

## Deferred structural improvement

An append-only JSONL event stream, reduced to a final report at process exit,
would remove shared mutable field ownership entirely. That is the preferred
long-term design, but it is intentionally deferred until after the current
300k-to-500k sweep is unblocked. The bounded fix here preserves the existing
consumer schema and existing checkpoint provenance.
