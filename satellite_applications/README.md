# Satellite applications

Applications of InfluPaint beyond the state-level influenza paper. These share the model and utilities in `influpaint/`.

| Directory | Contents |
| --- | --- |
| [metrocast/](metrocast/) | Metrocast data preparation, interactive forecasting, and calibration launcher |
| [RSV/](RSV/) | RSV data preparation and inpainting |

Use the research repository root as the working directory so package imports, CoPaint, and data paths resolve consistently.

## Metrocast

- `metrocast/1-gather_metrocast_datasets.py`: gather and prepare Metrocast training data.
- `metrocast/influpaint_metrocast_ipynb.py`: interactive forecasting notebook in Jupytext format.
- `metrocast/influpaint_metrocast_calib.py`: calibration entry point.
- `metrocast/metrocast_calib.run`: Slurm calibration array.

Submit from the repository root:

```bash
sbatch satellite_applications/metrocast/metrocast_calib.run
```

Metrocast source data remains under `Flusight/metrocast/`; training datasets remain under `training_datasets/`. Set the experiment, forecast date, input files, and cluster environment in the application before running it.

## RSV

- `RSV/RSV.qmd`: Quarto document for preparing RSV data.
- `RSV/rsv_inpaint.py`: inpainting script with its model and forecast configuration.

Run the Python entry point from the repository root:

```bash
python -m satellite_applications.RSV.rsv_inpaint
```

The Quarto document also uses repository-root data paths; render it with the repository root as the execution directory.
