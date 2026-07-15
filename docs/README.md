# Documentation (`docs/`)

Human-facing guides that explain *how the study is organized*, not executable pipeline code.
Pipeline code lives in `scripts/` and `src/`; committed numbers live in `results/paper/`.

## Files in this folder

| File | Audience | What it contains |
|---|---|---|
| [`paper_pipeline.md`](paper_pipeline.md) | Reviewers / reproducers | Stage-by-stage map of the canonical epoch-5 triage pipeline: which command runs, what it produces, where the committed snapshots sit, leakage rules, and what is optional vs required. |
| [`manuscript/`](manuscript/) | Manuscript authors | Verified environment and split notes (case counts, device/log provenance). |

## Recommended reading order

1. Root [`README.md`](../README.md) — research claim, main table, limitations.
2. This folder’s [`paper_pipeline.md`](paper_pipeline.md) — exact regeneration path.
3. [`configs/README.md`](../configs/README.md) — which YAML drives each stage.
4. [`scripts/README.md`](../scripts/README.md) — CLI wrappers in pipeline order.
5. [`src/README.md`](../src/README.md) — library layout; drill into `src/*/README.md` only when you need module detail.
6. [`results/paper/README.md`](../results/paper/README.md) — how to read committed tables without re-running GPU jobs.
7. [`extra/README.md`](../extra/README.md) — only if you care about RQ1 probing / editing / repair.

## What is *not* here

- Algorithmically frozen analysis logic → `src/analysis/`
- Runnable commands → `scripts/` and root `train*.py`
- Compact paper numbers → `results/paper/`

## Roadmap (documentation)

| Status | Item |
|---|---|
| Done | Pipeline map, split/environment notes, per-folder READMEs |
| Optional | Manuscript Methods draft linked from here once wording is frozen |
| Out of scope | Regenerating large `outputs_*` folders into git |
