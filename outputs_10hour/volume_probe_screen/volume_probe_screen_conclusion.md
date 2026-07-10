# Decoder2 Volume Probe Screening Conclusion

The probe direction **does not clearly outperform** matched random perturbations on mean |Δ tumor volume|.

## Overall (30 cases, |α|=1)

| Metric | Value |
|--------|------:|
| mean \|Δ volume\| probe | 3.82 vox |
| mean \|Δ volume\| random | 4.45 vox |
| probe / random ratio | 0.86 |
| probe percentile vs random | 0.47 |
| +α/−α opposite (actual volume) | 57% of cases |
| +α/−α opposite (analytical probe) | 100% of cases |

## By Dice bin

| bin | probe | random | ratio | percentile | opp. vol | opp. anal |
|-----|------:|-------:|------:|-----------:|---------:|----------:|
| low | 5.1 | 4.7 | 1.08 | 0.53 | 80% | 100% |
| medium | 3.3 | 5.1 | 0.65 | 0.42 | 50% | 100% |
| high | 3.0 | 3.6 | 0.86 | 0.47 | 40% | 100% |

**Interpretation:** High analytical opposite-sign rate with near-zero actual volume change indicates the probe axis moves in representation space but decoder2 edits do not propagate to segmentation volume.