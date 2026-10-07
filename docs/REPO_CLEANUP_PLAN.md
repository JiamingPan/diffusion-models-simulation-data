# Repo cleanup plan (2026-10-07)

State found on the Mac checkout (`/Users/apple/AI/Diffusion_model`, 1.4 GB):
19 modified tracked files, 8 deleted paper sections, about 100 untracked source
files that belong in git, and about 550 untracked scratch files that do not.
The Great Lakes checkout (`/home/jiamingp/diffusion_models_repo`) has this
week's executed notebooks and results and is where Claude Code runs. The two
checkouts have diverged; rsync was used to copy code one way, so neither is a
clean superset.

## Decision 1: one source of truth

Use git for code, not rsync. Great Lakes is where the work happens, so:

1. On Great Lakes, commit everything that belongs in the repo on a branch
   `agent-loop-setup`, push it (APPROVE PUSH).
2. On the Mac, commit the local modifications on a branch `mac-wip`, push it.
3. Merge `mac-wip` into `agent-loop-setup` on Great Lakes, resolve the few
   conflicts (notebooks: keep the GL executed versions; paper: see Decision 3),
   then merge to `main`.
4. From then on, `git pull` on the Mac. rsync only for files git ignores
   (figures, samples) and only Mac -> GL for things made on the Mac.

## Decision 2: what goes into git (commit)

* `scripts/*.py` untracked (about 70): all of them. They are the analysis and
  build scripts behind this week's results.
* `tests/test_*.py` untracked (21): all of them; run `pytest tests/` first and
  record the count in the commit message.
* `notebooks/*.ipynb` untracked (14) and `docs/*` (14): commit. Executed
  notebooks go under `notebooks/executed/` and are committed as they are the
  provenance record; strip nothing.
* `paper/`, `venue_drafts/`: commit the sources (`.tex`, `.bib`, figures under
  200 KB). Move `venue_drafts/` to `paper/venue_drafts/`.
* `simdiff_eval/` untracked (7): commit.
* The new agent-loop files: `CLAUDE.md`, `.claude/`, `scripts/ledger.py`,
  `experiments/`, `docs/agent_loop.md`, `docs/explainers/`.

## Decision 3: the deleted paper sections

`paper/ai4science_verification/sections/01..08_*.tex` are deleted in the Mac
working tree, and `main.tex` (272 lines, 31 section headings, no `\input`)
appears to have absorbed them. Before committing the deletion, confirm with
`git diff HEAD -- paper/ai4science_verification/main.tex | head -80` that the
content moved rather than vanished. If it did, commit as "paper: fold sections
into main.tex". If not, `git checkout -- paper/ai4science_verification/sections/`
restores them.

## Decision 4: what stays out of git (ignore or leave untracked)

Add to `.gitignore`:

```
/outputs/
/output/
/tmp/
/meeting_deck/build/
/meeting_deck/.codex-finalizer/
/src/__pycache__/
/Claude outputs/
/Neupris/
/eval/
/cosmo_diffusion_augment_pr/
/*.md.bak
coverage.xml
.coverage
```

`outputs/` (59 MB of figures and decks), `tmp/pdfs` (52 MB), `output/pdf`
(30 MB), `meeting_deck/build` (18 MB) are build products; the sources that
make them are already tracked or listed above.

## Decision 5: what to delete from the Mac disk (not from git)

These are ignored already and only take space; delete when convenient:

* `diffusion_pytorch_model.safetensors` (231 MB, April checkpoint)
* `diffusion.ipynb`, `diffusion_jp*.ipynb`, `inspect_run*.ipynb`,
  `my_diffusion.ipynb` at the repo root (April working notebooks, superseded by
  `notebooks/`)
* `configs_saved/`, `batch_scripts/` (pre-reorganisation)
* `.pytest_cache/`, `.ipynb_checkpoints/`, `.coverage`, `coverage.xml`

Nothing here is deleted by an agent. The user deletes after reading this list.

## Decision 6: top-level notes

`PROJECT.md`, `AstroAI_notes_summary.md`, `CODEX_PROMPT_dit_option.md`,
`spec.md`, `slack_to_nick_scaling.md`, `poster_presentation_script.md`: move to
`docs/notes/` and commit, or delete. They are small; keeping them under
`docs/notes/` costs nothing and preserves the history of how decisions were made.

## README

Add the section "How this project is run" (text in `docs/agent_loop.md`,
first two headings) after "Evaluation Framework". Keep the project title and
the opening paragraph as they are.
