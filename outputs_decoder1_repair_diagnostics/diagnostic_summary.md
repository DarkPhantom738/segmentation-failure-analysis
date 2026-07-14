# Decoder1 repair diagnostics

## Implementation checks
- unet_frozen: True
- decoder1_channels: 32
- probe_unit_norm: 1.0
- complete: True

## Classifier projection of probe `q = W @ d_probe`
- q = bg=0.1871, nec=0.1510, edema=-0.6117, enh=-0.0911
- edema-bg=-0.7988, edema-enh=-0.5207, edema-nec=-0.7628
- mean |edema-bg| for 5 random directions: 0.3137

## 1. Does the probe have meaningful oracle Dice headroom?
- mean/median/max oracle gain: 0.0192 / 0.0123 / 0.0762
- cases with gain≥0.02: 1/6; ≥0.05: 1/6
- Answer: NO

## 2. Can rank-1 architecture overfit one case by ≥0.05?
- Variant A improvement: 0.1604 (params=34)
- Answer: YES

## 3. Does a free direction outperform the probe?
- Variant B improvement: 0.1688
- Oracle-mask free-dir mean gain: 0.1572 vs oracle-mask probe 0.0000
- Answer: YES

## 4. Does K=4 multi-direction outperform rank-1?
- Variant C improvement: 0.1789 (params=260)
- Answer: YES

## 5. Does perfect oracle localization make the probe useful?
- Mean gain (oracle mask + probe): 0.0000
- Answer: NO

## 6. Does direct oracle logit correction show substantial headroom?
- Mean gain (oracle logit residual): 0.1733
- Answer: YES

## 7. Why did random gated directions tie the proposed method?
- Probe edema-vs-bg logit shift: -0.7988 (negative ⇒ global/local probe edits push *against* edema at the classifier).
- Random directions mean |edema-bg| shift: 0.3137
- Global probe oracle headroom is tiny on most cases, so gated edits at modest strength move Dice by ~noise; random gates with similar RMS therefore look tied.

## 8. Dominant limitation
- direction
- Decision-table case ≈ 2
- Key evidence: oracle mask + probe gain≈0, while oracle mask + free direction and oracle logit residual succeed (mean gains 0.157 / 0.173).
- Classifier projection shows the saved edema probe is anti-aligned with edema logits after W (q_edema strongly negative vs bg).

REPLACE PROBE DIRECTION
