# Marso-Hack-Berlin-2026

## SmolVLA Colab inference

Open `smalVLA_Project_COLAB_INFERENCE.ipynb` in a Colab T4 runtime and run the
cells in order. The notebook downloads the inference checkpoint from this
repository via Git LFS. By default it runs the Hard-task hybrid where the 30K
SmolVLA selects targets and retry/next macro primitives while a deterministic
Cartesian controller executes pick, transfer, and release. Set
`RUN_STANDARD_VLA=True` in the notebook to reproduce the original VLA-only
rollouts instead.

Checkpoint: `checkpoints/smolvla_all_recovery_30k`

The checkpoint was trained for 30,000 steps with `train_expert_only=true` and
`freeze_vision_encoder=true`. Local official-metric validation produced Easy
6/6, Medium 8/12, and Hard 1/18 sorted parcels (weighted score 42.78%).

The no-training hybrid in `hybrid/` produced Hard 2/6 on each of seeds 42, 43,
and 44 (6/18 total) in the local fixed-seed evaluation. The notebook records
the rollout videos and System-2 decision log to
`/content/vla_advisory_hybrid_eval/`.
