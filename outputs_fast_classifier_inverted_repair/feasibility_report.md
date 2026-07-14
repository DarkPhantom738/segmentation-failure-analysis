# Classifier-inverted repair feasibility report

**Verdict: NOT PROMISING**

## Split
- train: ['BraTS2021_01205', 'BraTS2021_00293', 'BraTS2021_00589', 'BraTS2021_01186', 'BraTS2021_01181', 'BraTS2021_01353', 'BraTS2021_01291', 'BraTS2021_01562', 'BraTS2021_01083', 'BraTS2021_00369', 'BraTS2021_00730', 'BraTS2021_01604']
- dev: ['BraTS2021_01511', 'BraTS2021_01529', 'BraTS2021_00299', 'BraTS2021_01018', 'BraTS2021_00413', 'BraTS2021_01306']
- test: ['BraTS2021_01464', 'BraTS2021_01245', 'BraTS2021_00539', 'BraTS2021_01180', 'BraTS2021_00165', 'BraTS2021_01552']

## Training
- device: `mps`
- steps logged: 100
- train runtime_s: 33.8
- best dev edema Dice: 0.7789
- tuned abstain threshold: 0.2
- tuned probe alpha: 0.5
- tuned logit bias: 0.5
- learned scale: 0.1195

## Method summary (test)
        method  mean_edema_dice  mean_fg_dice  mean_wt_dice  mean_precision  mean_recall  mean_enh_change  mean_nec_change  mean_runtime
D_logit_refine         0.833170      0.833843      0.894392        0.112299     0.938092      -288.851741      -239.045582      0.672670
    A_baseline         0.821193      0.821007      0.891964        0.000000     0.000000         0.000000         0.000000      0.145305
  C_logit_bias         0.820854      0.822590      0.889560        0.000000     0.000000       -50.305166       -52.721391      0.115807
    E_proposed         0.820369      0.820533      0.891328        0.104337     0.944338        -4.553899         2.049379      0.408102
B_global_probe         0.819459      0.819400      0.892538        0.000000     0.000000         1.708509        44.580629      0.173713

## Answers
1. Can the transition gate locate edema-related errors?
   - mean precision=0.104, recall=0.944; detects=YES
2. Which transitions are common on test (proxy for difficulty / prevalence)?
- bg_to_edema: total error voxels on test=6710
- nec_to_edema: total error voxels on test=2097
- enh_to_edema: total error voxels on test=1336
- edema_to_bg: total error voxels on test=4503
- edema_to_nec: total error voxels on test=124
- edema_to_enh: total error voxels on test=817
3. Does classifier-inverted decoder1 repair improve edema Dice?
   - mean Δ=-0.0008; cases improved=2/6
4. Does it beat direct logit bias?
   - proposed=0.8204 vs bias=0.8209 → NO
5. Does it beat learned logit-refinement?
   - proposed=0.8204 vs refine=0.8332 → NO
6. FN vs FP changes (baseline − proposed; positive ⇒ fewer errors):
   - ΔFN=+5.3, ΔFP=-16.5
7. Other tumor classes:
   - mean enhancing vol change=-4.55
   - mean necrosis vol change=+2.05
   - mean fg Dice drop=+0.0005
8. Result: **NOT PROMISING**

## Notes
- U-Net remained frozen (hash unchanged).
- Ridge probe used only as baseline B, not for proposed directions.
- Feasibility gates only; not clinical or publication claims.
