# Experiment ledger

`experiments/ledger.jsonl` is the single record of what was planned, run and
found. One JSON object per line, append-only; a later line for the same `run`
supersedes earlier ones. Agents and humans both write to it only through
`scripts/ledger.py`.

## Lifecycle

```
planned  --submit-->  submitted  --result-->  done
   |                      |
   +--------fail----------+---------------->  failed
```

* `plan`   records the question, the prediction (written before the run), the
  metric that decides it, the config path and its sha256, and the GPU estimate.
* `submit` records the Slurm job id. The `guard_sbatch.py` hook refuses any
  `sbatch` whose `RUN_NAME` has no `planned` entry, so a run cannot start
  without a prediction on record.
* `result` records the metrics file written by `scripts/evaluate_run.py` and a
  verdict (`supported`, `unsupported`, `mixed`) written after the critic has
  checked it.
* `fail`   closes a run that crashed or was abandoned, with a note.

## What counts as a metric

Only values produced by `scripts/evaluate_run.py` (novelty score G with PCA32 /
real-real 95th-percentile threshold, nearest-training cosine, draw-to-draw
cosine, P(k) ratio by scale band, probe median Omega_m bias and 68% coverage on
the 32 held-out cosmologies). No agent may define a new metric inside a run;
propose it as a change to `evaluate_run.py` with a test.

## Reading it

```
python scripts/ledger.py show              # latest status of every run
python scripts/ledger.py show --run NAME   # history of one run
```

Every number quoted in the paper or in a message to a collaborator must trace to
a `done` entry here and to the metrics file it points at.
