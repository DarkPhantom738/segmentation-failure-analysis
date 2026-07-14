# TTA disagreement (Task 7) — skipped

Flip TTA is already used to produce the retained hard masks and the frozen confidence features (mean entropy / max-prob summaries over averaged TTA probabilities). Computing pairwise flip–flip Dice / volume disagreement would require re-running sliding-window inference and storing per-flip hard segmentations for all 375 cases — a significant inference job, not a light post-hoc feature. Per the task brief, this section is skipped.
