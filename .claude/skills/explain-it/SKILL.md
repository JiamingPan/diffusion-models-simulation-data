---
name: explain-it
description: Explain a technical concept (model architecture, training, statistics, a result) as a clear visual HTML explainer page with diagrams, equations and interactive toggles, plus a short plain-language summary. Use when the user asks to explain, understand or visualize something, or types /explain-it.
---

# explain-it

Goal: the user understands the concept quickly and correctly. Prefer pictures and equations over long text.

## 1. Get the facts right first
- Read the actual code, configs and library source before explaining. Do not explain from memory when the repo can confirm it.
- Use the project's real numbers (shapes, channel counts, hyperparameters, results). Label anything that is a simplification.

## 2. Writing style (about 80% of ASD-STE100 Simplified Technical English)
- Short sentences (aim for 20 words or fewer). One idea per sentence. Active voice.
- One word for one meaning. Define each technical term the first time you use it.
- Use lists and tables for steps and comparisons. Do not use filler.

## 3. Build an HTML explainer page
Write one self-contained `.html` file to a fresh timestamped path:
`results/explainers/<topic>_<YYYYMMDD_HHMMSS>.html` (never overwrite an existing file).

The page must have:
- **A diagram** (inline SVG) of the structure or flow, with the real shapes and sizes labeled.
- **Equations** rendered with KaTeX from cdnjs (`katex.min.css`, `katex.min.js`, `auto-render.min.js`). Define every symbol.
- **Interactivity where it helps understanding**: toggles that compare variants (for example baseline vs new), highlights, or a small plot driven by JS.
- **A worked example** with concrete numbers.
- **A "key points" box** at the end (3 to 6 bullets).
- Light and dark color themes (CSS variables on `:root`, `prefers-color-scheme`), readable at phone width, no horizontal scroll.
- Only these external resources: cdnjs.cloudflare.com scripts/styles. Everything else inline.

## 4. Deliver it
- The user's terminal paste picks up junk from the Claude side panel, so give download commands the user can TYPE: short lines, no globs that need quoting, e.g.
  `cd ~/Downloads` then `scp jiamingp@greatlakes.arc-ts.umich.edu:diffusion_models_repo/<path> .` then `open <file>`.
- Publishing as a claude.ai Artifact link sends content off the cluster: offer it, and publish only after the user says APPROVE SEND.
- In the chat reply, give a 3 to 6 line plain-language summary and the file path. Do not repeat the whole page.
