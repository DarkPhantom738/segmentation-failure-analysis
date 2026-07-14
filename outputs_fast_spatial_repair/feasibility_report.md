# Fast spatial edema-repair feasibility report

**Verdict: NOT PROMISING**

## Split (train partition only; seed=feasibility_seed)
- train: ['BraTS2021_01205', 'BraTS2021_00293', 'BraTS2021_00589', 'BraTS2021_01186', 'BraTS2021_01181', 'BraTS2021_01353', 'BraTS2021_01291', 'BraTS2021_01562', 'BraTS2021_01083', 'BraTS2021_00369', 'BraTS2021_00730', 'BraTS2021_01604']
- dev: ['BraTS2021_01511', 'BraTS2021_01529', 'BraTS2021_00299', 'BraTS2021_01018', 'BraTS2021_00413', 'BraTS2021_01306']
- test: ['BraTS2021_01464', 'BraTS2021_01245', 'BraTS2021_00539', 'BraTS2021_01180', 'BraTS2021_00165', 'BraTS2021_01552']

## Runtime / device
- device: `mps`
- training runtime_s: 78.8
- optimization steps: 75
- trainable parameters: 66
- learned softplus scale: 0.663022
- cosine(d_learned, d_initial): 0.999047

## Method summary (held-out test)
         method  mean_edema_dice  mean_fg_dice  mean_wt_dice  mean_abs_edema_vol_err  mean_enh_vol_change  mean_nec_vol_change  mean_runtime
     F_proposed         0.821855      0.821621      0.893132             1512.500000           -36.520304           -35.321996      1.224113
    E_random_03         0.821454      0.820993      0.892753             1386.666667           -10.653015            -8.462514      2.402668
    E_random_02         0.821440      0.821068      0.891958             1262.666667            -1.077372            -5.005117      2.392704
    E_random_01         0.821312      0.821504      0.891261             1193.500000           -18.726949            -9.385030      2.414043
    E_random_04         0.821264      0.820900      0.892216             1308.000000             9.240255            -4.418986      2.392089
     A_baseline         0.821193      0.821007      0.891964             1256.500000             0.000000             0.000000      1.324854
   D_logit_bias         0.820854      0.822590      0.889560             1029.500000           -50.305359           -52.721436      1.317771
    E_random_00         0.820316      0.820522      0.891353             1220.666667            -5.178507            15.101593      2.393226
 B_global_probe         0.819459      0.819400      0.892538             1437.000000             1.708313            44.580597      1.351469
C_global_output         0.814666      0.813023      0.883022             1898.833333          -252.728394          -218.625031      1.320206

## Per-case baseline vs proposed (edema Dice)
- BraTS2021_01464: baseline=0.6677, proposed=0.7056, Δ=+0.0379
- BraTS2021_01245: baseline=0.6961, proposed=0.6944, Δ=-0.0016
- BraTS2021_00539: baseline=0.8699, proposed=0.8530, Δ=-0.0168
- BraTS2021_01180: baseline=0.8195, proposed=0.8108, Δ=-0.0087
- BraTS2021_00165: baseline=0.9425, proposed=0.9427, Δ=+0.0002
- BraTS2021_01552: baseline=0.9315, proposed=0.9245, Δ=-0.0070

## Off-target soft volume changes (proposed, mean)
- enhancing: -36.52
- necrosis: -35.32

## Notes
- Feasibility gates only; not a statistical or clinical claim.
- Validation (375) cases were never used.