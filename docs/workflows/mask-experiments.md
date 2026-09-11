# 7. Reconstruct missing observations

You will use **the selected i868 checkpoint and different observation masks** to obtain **reconstructed epidemic trajectories**, showing that the same model can fill gaps beyond future weeks.

[![Reconstruction results for missing locations, missing weeks, and checkerboard masks](../assets/paper/figure-4-reconstruction.png)](../assets/paper/figure-4-reconstruction.png)

**Figure 4 — Reconstruction with different observation masks.** Each inset shows the entries supplied to the model in green and the hidden entries in red. Black curves show observed hospitalizations; colored lines and fans show reconstructed medians and predictive quantiles. The same trained model handles missing states, midseason gaps, missing early weeks, and checkerboard patterns by changing the mask at sampling time.

!!! tip "Skip this step with the reproduction archive"

    Use `influpaint-paper/influpaint_paper_reproduction_data/forecasts/masks/`. Each experiment directory contains `fluforecasts_ti.npy`, `mask.npy`, and `ground_truth.npy`. For example, `influpaint-paper/influpaint_paper_reproduction_data/forecasts/masks/missing_nc_season2023/fluforecasts_ti.npy` contains the North Carolina reconstruction ensemble.


## Change the observed region

A forecasting mask hides future weeks. These experiments instead hide whole locations, an internal time interval, past weeks, or scattered blocks. CoPaint conditions on the remaining entries and generates the missing portion using the same selected model.

```bash
sbatch main_training/mask_experiments.run
```

The launcher runs `python -m main_training.mask_experiments` from the repository root. It requests one GPU, 64 GB RAM, and ten hours on the original Longleaf configuration. The module retrieves i868 from `paper-2025-07-22_training` and uses `celebahq_noTTJ5`.

## Compare reconstruction with withheld truth

| Archived directory under `forecasts/masks/` | Hidden entries |
| --- | --- |
| `missing_half_subpop_season2023` | Approximately half the locations |
| `missing_nc_season2023` | North Carolina |
| `missing_il_season2023` | Illinois |
| `missing_midseason_biggap_season2023` | A midseason interval |
| `missing_past_season2023` | The early season |
| `missing_checkerboard_4x4_season2023` | Alternating four-week by four-location blocks |

In `mask.npy`, 1 means observed and 0 means hidden. The mask and saved truth have shape `(1,64,64)`; reconstruction arrays have shape `(512,1,64,64)`. Figure 4 compares the generated trajectories against the saved, untransformed truth for the 2023–2024 season.

[Previous: 6. Score forecasts](evaluation.md) · [Next: 8. Reproduce paper figures](paper-figures.md)
