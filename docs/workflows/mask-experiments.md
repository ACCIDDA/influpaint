# 7. Reconstruct missing observations

The objective is to explore what the selected model can infer when observations are missing in patterns other than a forecast's future weeks. Starting with the checkpoint and CoPaint setting selected in step 6, hide whole states, a stretch of weeks, or alternating blocks, then generate trajectories conditioned on the entries that remain visible.

The workflow is the same as forecasting: prepare data and a mask, sample with fixed model weights, and compare the generated values with the hidden observations. Only the observation pattern changes.

[![Reconstruction results for missing locations, missing weeks, and checkerboard masks](../assets/paper/figure-4-reconstruction.png)](../assets/paper/figure-4-reconstruction.png)

**Figure 4 — Reconstruction with different observation masks.** Each inset shows the entries supplied to the model in green and the hidden entries in red. Black curves show observed hospitalizations; colored lines and fans show reconstructed medians and predictive quantiles. The same trained model handles missing states, midseason gaps, missing early weeks, and checkerboard patterns by changing the mask at sampling time.

!!! tip "Use saved results from the Zenodo archive"

    Use `influpaint-paper/influpaint_paper_reproduction_data/forecasts/masks/`. Each experiment directory contains `fluforecasts_ti.npy`, `mask.npy`, and `ground_truth.npy`. For example, `influpaint-paper/influpaint_paper_reproduction_data/forecasts/masks/missing_nc_season2023/fluforecasts_ti.npy` contains the North Carolina reconstruction ensemble.

## 1. Define which observations to hide

A forecasting mask hides future weeks. These experiments instead hide whole locations, an internal time interval, past weeks, or scattered blocks. CoPaint conditions on the remaining entries and generates the missing portion using the same selected model.

Open `main_training/mask_experiments.py` to inspect its season, model selection, and mask definitions. Choose the experiment and scenario for your checkpoint, and set the CoPaint configuration you want to use. Each mask defines the evidence available to the sampler; retaining the original truth separately lets you evaluate the withheld entries afterward.

## 2. Run the reconstruction experiments

Adapt the Python environment and Slurm resources in the launcher, then submit from the repository root:

```bash
sbatch main_training/mask_experiments.run
```

The launcher runs `python -m main_training.mask_experiments` from the repository root. It requests one GPU, 64 GB RAM, and ten hours on the original Longleaf configuration. The module retrieves i868 from `paper-2025-07-22_training` and uses `celebahq_noTTJ5`.

## 3. Compare reconstruction with withheld truth

| Archived directory under `forecasts/masks/` | Hidden entries |
| --- | --- |
| `missing_half_subpop_season2023` | Approximately half the locations |
| `missing_nc_season2023` | North Carolina |
| `missing_il_season2023` | Illinois |
| `missing_midseason_biggap_season2023` | A midseason interval |
| `missing_past_season2023` | The early season |
| `missing_checkerboard_4x4_season2023` | Alternating four-week by four-location blocks |

In `mask.npy`, 1 means observed and 0 means hidden. The mask and saved truth have shape `(1,64,64)`; reconstruction arrays have shape `(512,1,64,64)`. Figure 4 compares the generated trajectories against the saved, untransformed truth for the 2023–2024 season. Look at the hidden regions when assessing reconstruction: how much does the model recover from neighboring weeks or other locations, and how does uncertainty widen where evidence is sparse? Carry the samples, masks, and truth together into the figure workflow.

[Previous: 6. Score forecasts](evaluation.md) · [Next: 8. Reproduce paper figures](paper-figures.md)
