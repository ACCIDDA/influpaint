# Paper training and calibration

Run all commands from the research repository root, with the InfluPaint environment active. The numbered explanation is in [the Material walkthrough](../docs/index.md); [runs.txt](runs.txt) collects the commands in execution order.

| File | Purpose |
| --- | --- |
| `train.run` | Longleaf array requesting 17 candidate training scenarios for `paper-2025-07-22_training` |
| `paper_runs_2025-07-22/scenarios.txt` | Preserved scenario definitions; option ordering determines numeric IDs |
| `paper_runs_2025-07-22/inpaint_jobs_paper-2025-07-22.txt` | Unmodified historical manifest: 15 finished models × 29 dates × 3 CoPaint configurations = 1,305 jobs |
| `inpaint_array_paper-2025-07-22.run` | Launcher reconstructed from the current batch generator for the preserved manifest; the original generated launcher was unavailable |
| `generated/` | Default destination for new manifests and launchers from `influpaint.batch.generate_inpainting_jobs` |
| `mask_experiments.py`, `mask_experiments.run` | Python module and Slurm launcher for additional conditioning-mask experiments |
| `runs.txt` | Commands for training, forecasting, scoring, reconstruction, and figure reproduction |

The Slurm files contain the original Longleaf partitions, resources, and Python environment path. Adapt these to your cluster before submission. Submit from the repository root: Python imports, CoPaint, data paths, and MLflow tracking use that working directory. For example, `sbatch main_training/train.run` and `python -m main_training.mask_experiments`.

The saved manifest references original MLflow training run IDs, including i868 run `fabf8cf1fbf2464380b1747e4974a1d7`. Reusing it requires the corresponding MLflow store and artifacts. After retraining, generate a new manifest under `main_training/generated/`; retain the historical manifest unchanged. Training and forecast implementations remain in the reusable `influpaint.batch` package.

The paper configuration uses the July 17, 2025 NetCDF files. Restore them from `influpaint-paper/influpaint_paper_reproduction_data/datasets/training/`. The reproduction archive contains the selected checkpoint and forecasts, not all candidate checkpoints or their MLflow store. See its [README](../influpaint-paper/influpaint_paper_reproduction_data/README.md) for the exact paths and provenance.
