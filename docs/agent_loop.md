# How this project is run: the agent loop

The experiments in this repository are planned, prepared and checked by coding
agents (Claude Code) under a fixed loop with human gates. The agents do not
decide what is true; they produce planned runs, files and checks that a person
approves and reads.

## The loop

1. **Spec.** `CLAUDE.md` holds the environment rules, what is established (with
   file paths), what is not, and the ordered list of open items. Every agent
   session reads it first.
2. **Plan.** The `planner` agent proposes one experiment: question, prediction
   written before the run, the metric that decides it, exact inputs, and the
   result files it will be compared against. It writes a `planned` entry to
   `experiments/ledger.jsonl`.
3. **Prepare and run.** The `executor` agent writes configs and Slurm files to
   new paths, runs CPU analyses itself, and stops at the gate for anything on a
   GPU or anything destructive. The user replies `APPROVE RUN`, `APPROVE EDIT`,
   `APPROVE PUSH` or `APPROVE DELETE`. A pre-command hook refuses `sbatch` for
   any run without a planned ledger entry, and refuses `rm`, `scancel`, `pip`,
   `conda`, `--overwrite` and `git push` outright.
4. **Evaluate.** `scripts/evaluate_run.py` computes the project's fixed metrics.
   Agents may not invent metrics inside a run.
5. **Check.** The `critic` agent takes a claim (a paper sentence, a Slack
   message, a ledger verdict) and checks every number against a file and every
   causal word against the alternatives. It writes the defensible version.
6. **Record.** The result and verdict go into the ledger. Nothing is quoted
   outside the repo unless it traces to a `done` ledger entry.

## Two things the loop caught

* The training recipe was believed to have no augmentation; the manifests said
  so. A runtime audit (`scripts/audit_training_map_augmentation.py`) built the
  dataset exactly as training does and fetched the same index eight times to
  prove it, before an augmentation sweep was designed around the assumption.
* After the two-parameter conditioning runs, the proposed reading was
  "memorization gives unbiased medians with overconfident error bars". The
  critic showed from the files that copying stops by N = 1024 (draw cosine
  0.06) while the low-Omega_m bias persists, so the bias is a separate
  generator-quality effect and the claim was rewritten before it reached the
  collaborator.

## Files

```
CLAUDE.md                     spec: rules, established results, open items
.claude/agents/planner.md     proposes runs, writes planned ledger entries
.claude/agents/executor.md    prepares and runs under the approval gate
.claude/agents/critic.md      checks claims against files
.claude/hooks/guard_sbatch.py pre-command hook: ledger check, hard blocks
.claude/settings.json         permissions and hook wiring
scripts/ledger.py             the only writer of the ledger
experiments/ledger.jsonl      the record
experiments/LEDGER.md         ledger format and lifecycle
docs/explainers/              plain-language pages the agents produce on request
```

## What the agents do not do

They do not submit GPU jobs, delete files, push to GitHub or send messages
without an explicit approval. They do not write the paper's claims; they draft,
the critic checks, the authors decide.
