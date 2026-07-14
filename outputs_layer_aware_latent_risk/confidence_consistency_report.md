# Confidence export consistency report

- Checkpoint: `outputs_10hour/checkpoints/checkpoint_epoch_005.pt` (epoch 5)
- Mode: **TTA × 8** flips, sliding-window patch `[64,64,64]`, overlap `0.25`
- Preprocessing: checkpoint-embedded settings + `outputs_10hour/cache`
- Data root used: `data/BraTS2021_Training`
- Compared against retained hard masks: `*_pred_tta.npy` from `failure_metrics.csv`

## Result

| Check | Value |
|---|---|
| Cases | 375 / 375 |
| Exact argmax match vs saved TTA mask | **375 / 375** |
| Min voxel agreement | 1.0 |
| Full prob/entropy volumes written | **none** |
| Output | `case_level_confidence_features.csv` (~142 KB) |

Confidence features are safe to pair with existing Dice / failure labels.
