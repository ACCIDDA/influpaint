# 9. Generate operational forecasts

Use **`run_operational_flusight.py`**, the interactive forecasting notebook, to condition a trained model on the latest reported hospitalizations and export **FluSight CSVs, forecast PDFs, and sampled trajectories**. The example below comes from the early 2025–2026 season, with a forecast reference date of **November 29, 2025**.

## Inspect the observations and mask

[![The operational notebook's four-panel mask diagnostic for November 29, 2025](../assets/operational/2025-11-29-observation-mask.png)](../assets/operational/2025-11-29-observation-mask.png)

**Early-season observation mask.** This recreates the notebook's first graph, `gt1.plot_mask()`, using a surveillance snapshot from November 26, 2025, with observations through November 22. Rows are weeks and columns are locations on the padded 64 × 64 image grid. The mask is 1 for observed history and 0 for future weeks; its upper band is pink and its lower band is purple. The dashed line marks the forecasting boundary. Red entries in the data panels are missing values, while the extra rows and columns are padding. The last two panels overlay the same mask on the data and reference data; both use the same archived snapshot here, as in the notebook's `nogit=True` workflow.

Unlike [the reconstruction experiments](mask-experiments.md), the operational mask leaves the observed history available and hides the future. Inspect this graph before sampling to confirm the season, available observations, and forecast boundary.

!!! tip "Inspect a saved operational forecast"

    The example report and its preview are included on this page, so no model run is needed to inspect them. The original November 29 files are under `Flusight/2025-2026/Flusight 2025-2026/2025-11-29/` in the local research archive. These are separate from the paper reproduction archive's earlier operational submissions in `forecasts/operational/`.

## Open the notebook

Complete [environment setup](../getting-started/installation.md) and restore the [i868 training dataset](build-training-datasets.md) and trained checkpoint. Run from the research repository root. Open `run_operational_flusight.py` as a Jupytext notebook in your editor, or create an unexecuted notebook copy:

```bash
jupytext --to notebook run_operational_flusight.py --output run_operational_flusight.ipynb
jupyter lab run_operational_flusight.ipynb
```

Select **Python (diffusion_torch)** as the kernel. Run the cells interactively, setting the configuration before loading the model or starting sampling.

## Choose the model and forecast date

The notebook uses scenario **868**, **512 samples**, and CoPaint configuration **`celebahq_noTTJ5`**. Set `forecast_date` to the intended forecast reference date; the checked-in notebook contains a fixed example date that must be changed for a new run.

To load the selected checkpoint from the reproduction archive, set these variables in the configuration cell:

```python
scenario_id = 868
config_name = "celebahq_noTTJ5"
batch_size = 512
experiment_name = None
run_id = None
model_path = (
    "influpaint-paper/influpaint_paper_reproduction_data/"
    "i868::m_U500cRx1224::ds_30S70M::tr_Sqrt::ri_No::3000.pth"
)
```

The model/dataset cell still needs the scenario's training data to create its transformations. Alternatively, the notebook can load a finished i868 run from the local `paper-2025-07-22_training` MLflow experiment when that experiment and its artifacts are available.

## Update the data and run inpainting

Update the local source checkouts before preparing ground truth:

```bash
./update-data.sh
```

`GroundTruth.for_flusight` reads the local FluSight hospitalization data. For the 2025–2026 season, the current reader uses `Flusight/2024-2025/FluSight-forecast-hub-official/target-data/target-hospital-admissions.csv`; the directory name does not limit the seasons in that file. The notebook passes `nogit=True`, so `data_date` does not select a historical Git revision or download fresh data.

Run the ground-truth cells and inspect the mask shown above. Then run the sampler cells: they transform the observations, pass the data and mask to `O_DDIMSampler`, generate 512 trajectories, and transform the results back to hospitalization counts. The notebook then plots national and state forecasts.

## Review an example output

[![Alabama full-season and short-horizon forecasts from the saved November 29, 2025 report](../assets/operational/2025-11-29-forecast-alabama.png)](../assets/operational/2025-11-29-forecast-50-95.pdf)

**Saved operational forecast for Alabama, November 29, 2025.** This is an excerpt from the original `plot50-95.pdf` in `Flusight`, showing the full season at left and the forecast window at right. Blue curves show medians and shaded bands show the 50% and 95% prediction intervals. Black points show available observations, gray dashed curves show earlier seasons, and vertical lines mark the forecast boundary and horizon in the full-season view.

[Open the full original forecast PDF, including national and state panels](../assets/operational/2025-11-29-forecast-50-95.pdf)

## Export the forecast

The export cells choose the next Saturday from the current date as `submission_date`, refresh the ground-truth object from the local data, and write to `operational_output/<submission_date>/`. For a live run, keep the sampling date, observations, and submission reference date consistent. To replay a historical example, use its archived surveillance snapshot and explicitly set the export date and ground-truth cutoff to match; the default export cells use today's date even if `forecast_date` is historical.

With the notebook's default prefix `UNC_IDD-InfluPaint`, the outputs are:

| Output | Contents |
| --- | --- |
| `<date>-UNC_IDD-InfluPaint.csv` | Location and national quantiles for the FluSight targets |
| `UNC_IDD-InfluPaint-<date>-plot50-95.pdf` | Median forecasts with 50% and 95% intervals |
| `UNC_IDD-InfluPaint-<date>-plotall.pdf` | Median forecasts with the full set of quantile bands |
| `<date>_fluforecasts_raw.npy` | Samples on the transformed model scale |
| `<date>_fluforecasts_transformed_inv.npy` | Sampled trajectories on the hospitalization scale |
| `<date>_forecasts_national.npy` | National trajectories aggregated from the samples |

The array files are written when `save_raw_arrays=True`. Review the plots and exported dates before using the CSV as a submission. Running the notebook creates local files; submission to FluSight is a separate action.

## Figure sources

The mask uses `target-data/target-hospital-admissions.csv` from revision `aa25b8d92c06241d7e50dfb15b80fbc04f6483d5` in the local `Flusight/2024-2025/my-hub-fork-for-submissions` checkout. It is reconstructed from that archived surveillance snapshot, not extracted from a saved notebook output. The PDF is copied unchanged from `Flusight/2025-2026/Flusight 2025-2026/2025-11-29/UNC_IDD-InfluPaint_celebahq_noTTJ5-2025-11-29-plot50-95.pdf`; the inline preview crops its Alabama row.

Regenerate these documentation assets from the local source files with:

```bash
python -m main_training.render_operational_story
```

This calls the notebook's mask plotting method and uses Poppler's `pdftoppm` for the PDF preview. It does not generate new forecasts.

[Previous: 8. Reproduce the paper figures](paper-figures.md) · [Back to contents](start-here.md)
