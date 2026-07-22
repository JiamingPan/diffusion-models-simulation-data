# Public README Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the narrow project summary with a concise public overview of the repository's current modeling, evaluation, and Great Lakes workflow.

**Architecture:** Keep `README.md` as the single public entry point and link to existing notebooks, scripts, templates, and detailed notes instead of duplicating them. Describe established capabilities directly and mark DiT depth and capacity studies as exploratory.

**Tech Stack:** Markdown, Python, PyTorch, `cosmo_diffusion`, diffusers, Slurm

## Global Constraints

- Do not publish unpublished numerical results, run IDs, private paths, account names, or checkpoint locations.
- Do not present exploratory DiT-depth or capacity-scaling results as settled.
- Do not claim that one diagnostic alone proves scientific validity.
- Use plain technical language and short paragraphs.
- Keep unrelated working-tree changes untouched.

---

### Task 1: Rewrite the public project overview

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: Existing public paths under `configs/templates/`, `scripts/`, `scripts/slurm/`, `simdiff_eval/`, `notebooks/`, and `docs/`.
- Produces: A public entry point with Overview, What This Repository Contains, Evaluation Framework, Workflows and Repository Map, Setup, Research Status, and Acknowledgements sections.

- [x] **Step 1: Replace the current narrow introduction**

Describe the project as a study of unconditional and conditional diffusion models for two-dimensional CAMELS IllustrisTNG neutral-hydrogen fields. Name UNet and DiT backbones, but avoid reporting preliminary numerical comparisons.

- [x] **Step 2: Document the implemented capabilities**

Cover model/config preparation, training and checkpoint continuation, DPM-Solver sampling, pixel/PCA/SSCD nearest-neighbor diagnostics, cross-run reproducibility, one-point distributions, power spectra, and real-data-trained parameter encoders.

- [x] **Step 3: Add concrete repository and workflow pointers**

Link the overview to existing directories and representative notebooks. Keep cluster instructions portable and avoid account-specific commands or paths.

- [x] **Step 4: State research status conservatively**

Separate established evaluation infrastructure from active studies of DiT depth, model capacity, EMA, and augmentation. State that novelty, physical statistics, and conditional calibration answer different questions.

### Task 2: Verify the public README

**Files:**
- Verify: `README.md`
- Verify: `docs/superpowers/specs/2026-07-22-readme-refresh-design.md`

**Interfaces:**
- Consumes: Markdown links and paths introduced in Task 1.
- Produces: A clean, self-consistent README diff ready to commit.

- [x] **Step 1: Check referenced local paths**

Run a path audit over all backtick-delimited repository paths used in `README.md` and confirm every path exists.

- [x] **Step 2: Check for private details and malformed Markdown**

Run:

```bash
rg -n '/Users/|/home/|/scratch/|JobID|SBATCH -A|jiamingp@' README.md
git diff --check -- README.md
```

Expected: no private-path matches and no whitespace errors.

- [x] **Step 3: Review the scoped diff**

Run:

```bash
git diff -- README.md
git status --short
```

Expected: the README contains only the approved public-facing rewrite, and unrelated dirty files remain unchanged.

- [x] **Step 4: Commit the README and plan**

```bash
git add README.md docs/superpowers/plans/2026-07-22-readme-refresh.md
git commit -m "Refresh diffusion research overview"
```
