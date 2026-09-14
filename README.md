# Marso-Hack-Berlin-2026

## SmolVLA Colab inference

Open `smalVLA_Project_COLAB_INFERENCE.ipynb` in a Colab T4 runtime and run the
cells in order. The notebook downloads the inference checkpoint from this
repository via Git LFS and evaluates Easy, Medium, and Hard RGB rollouts.

Checkpoint: `checkpoints/smolvla_all_recovery_30k`

The checkpoint was trained for 30,000 steps with `train_expert_only=true` and
`freeze_vision_encoder=true`. Local official-metric validation produced Easy
6/6, Medium 8/12, and Hard 1/18 sorted parcels (weighted score 42.78%).
