# Results (`results/`)

Place for **committed, reviewer-facing numeric artifacts**. The only subtree intentionally tracked in git is [`paper/`](paper/README.md).

## Layout

```text
results/
  paper/          ← canonical tables + figures for the manuscript / root README
    triage_20260712/
    method_validation/
    consistency/
    *.csv
```

Anything regenerated into top-level `outputs_*` folders is local/ephemeral (gitignored). Prefer copying a curated subset into `results/paper/` only when deliberately updating the public snapshot.

## What readers should do

| Goal | Action |
|---|---|
| Inspect the paper numbers | Open `paper/README.md` and the CSVs/figures there |
| Regenerate a full run | Follow `docs/paper_pipeline.md`; write to `outputs_*` |
| Compare a new experiment | Keep new directories outside `paper/` until review |

## Roadmap

| Status | Item |
|---|---|
| Done | Compact epoch-5 triage + validation snapshot |
| Optional | Add compact multi-seed `converged_seed*/` summaries once seed suite finishes |
| Never | Commit full softmax volumes, checkpoints, or embedding matrices |
