# subhanga

This folder contains Subhanga's implementation work for Assignment 2.

## Part 1 (CNN)

- Model: `WeatherResNetGAP` (Global Average Pooling head). Use `out_dim=6` for the baseline (six regression targets) or `out_dim=7` for the ablation (six regressors + one binary logit). See `REG_HEAD_DIM` in `subhanga/models.py`.
- Training script: `subhanga/train.py`
- Checkpoints: baseline under `checkpoints/subhanga/`; 7-feature ablation under `checkpoints/subhanga_7feat/` (see Slurm below).

### 7-feature ablation (multitask, baseline-fair selection)

- Adds `binary_label` at \(t+\)lead as a seventh output with `BCEWithLogitsLoss` (class-balanced `pos_weight` on the train split). **Hyperparameters match the 6-feature run** (`--epochs 10`, `--batch-size 16`, `--base-channels 16`, default `--lr` / `--weight-decay`).
- **`best.pt` is still chosen by validation regression MSE only** (same metric as the vanilla model), so you can compare runs fairly. Logs prefix epochs with `[7feat]` and print auxiliary BCE for monitoring only.

```bash
# HPC (GPU job — same partition/resources as train_subhanga.slurm)
sbatch subhanga/train_subhanga_7feat.slurm

# Or manually:
python -m subhanga.train \
  --dataset-dir /cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset \
  --checkpoint-dir /cluster/tufts/c26sp1cs0137/supadh03/SkyOracle/checkpoints/subhanga_7feat \
  --epochs 10 --batch-size 16 --base-channels 16 \
  --multitask7
```

**Status:** This 7-feature version **is on the Tufts HPC GPU queue** when submitted (jobs often stay **pending** a long time under shared-cluster priority). For the **submitted report, slides, and primary results**, we are **moving ahead with Darius’s best model** (the main `WeatherCNN` / `checkpoints/best_model.pt` narrative in the root `README.md`). **If the 7-feature run completes before Monday’s presentation**, we will **discuss those findings** (e.g. validation regression MSE vs. the 6-output baseline) **if time allows**.

### Train (on HPC)

```bash
python -m subhanga.train \
  --dataset-dir /cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset \
  --checkpoint-dir /cluster/tufts/c26sp1cs0137/supadh03/SkyOracle/checkpoints/subhanga \
  --epochs 10
```

### Evaluate (on HPC)

1. Set `MODEL_NAME = "subhanga"` in `evaluation/evaluate.py`.
2. Point the evaluator to your checkpoint:

```bash
export SKYORACLE_CHECKPOINT="/cluster/tufts/c26sp1cs0137/supadh03/SkyOracle/checkpoints/subhanga/best.pt"
python evaluation/evaluate.py
```

