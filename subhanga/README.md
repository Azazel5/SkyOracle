# subhanga

This folder contains Subhanga's implementation work for Assignment 2.

## Part 1 (CNN)

- Model: `WeatherResNetGAP` (Global Average Pooling head, outputs 6 continuous targets)
- Training script: `subhanga/train.py`
- Checkpoints: by default saved under `checkpoints/subhanga/` as `best.pt` and `last.pt`

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

