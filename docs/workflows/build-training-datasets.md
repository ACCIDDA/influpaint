# 2. Create training data

The objective is to turn the source archive from step 1 into **complete training seasons with different mixtures of surveillance and simulated data**. This lets you ask whether learning mainly from observed epidemics, mainly from simulations, or from a blend of both gives better forecasts.

Use `dataset_creation/2-build_training_flu_datasets_ipynb.py` to read `Flusight/flu-datasets/all_datasets.parquet`, choose the source proportions, fill out the week/location grid, and save four NetCDF datasets. Each training example becomes an image: weeks run down the rows, locations run across the columns, and pixel values describe epidemic intensity.

Open the Jupytext script as notebook cells and use the repository root as the working directory. Read `Flusight/flu-datasets/all_datasets.parquet` first. Run the cells in order: inspect source scales, set the mixing recipes, build frames, then save and plot the resulting datasets. The notebook writes `training_datasets/TS_<mixture>_<today>.nc`; note those filenames because you will select them in the training configuration in step 3.

## 1. Compare source scales

The notebook sums available locations for each `(datasetH1, datasetH2, fluseason, sample, season_week)` and takes the peak across weeks. This exposes differences in source magnitude before mixing.

![Peak distributions from the four source families](../assets/notebook-stories/02-source-peaks.png)

*The notebook's peak calculation rendered from the archived Parquet table. Each histogram uses its own scale. Summed available locations do not necessarily provide equal geographic coverage across sources.*

FluView frames use `to_scale=True`. The mixer draws target peak magnitudes from the `SMH_R4-R5` peak distribution to rescale those frames while retaining their epidemic shapes.

## 2. Mix and complete season frames

The `DATASET_GRIDS` dictionary is where you define the training mixtures. `S` means surveillance (FluView and FluSurv), and `M` means modeled trajectories (flepiMoP and SMH). A recipe either gives a `multiplier` for each included family or gives each family a target `proportion` and a shared `total`. The mixed recipes request 3,000 frames; the mixer converts these targets into replication counts, so realized sizes and proportions can differ.

To change the balance, edit the proportions in `DATASET_GRIDS` before running the dataset-building loop. For example, `30S70M` allocates 15% to FluView, 15% to FluSurv, 5% to flepiMoP, and 65% to SMH. The table below gives the supplied recipes and the sizes of the saved example datasets.

`influpaint/datasets/mixer.py` turns each source season/trajectory into a frame with weeks 1–53 and all 50 states plus DC. With `fill_missing_locations="random"`, missing locations are completed using other season data, and `origin` records where the curves came from. `to_scale=True` rescales FluView frames using the peak distribution above. The converter then places every frame on the same ordered array grid.

| Mixture | Notebook recipe | Saved samples |
| --- | --- | ---: |
| `100S` | FluView and FluSurv, each with multiplier 26 | 520 |
| `70S30M` | 37% FluView, 33% FluSurv, 5% flepiMoP, 25% SMH | 3,332 |
| `30S70M` | 15% FluView, 15% FluSurv, 5% flepiMoP, 65% SMH | 3,223 |
| `100M` | flepiMoP and SMH, each with multiplier 1 | 1,240 |

![Realized sizes of the four archived mixtures](../assets/notebook-stories/02-mixture-sizes.png)

*Counts read from the exact July 17 NetCDF files, grouped by each frame's `main_origins` attribute. Main origin summarizes frame provenance; filled locations can have different origins. The percentages are mixing targets, not guarantees of exact realized proportions. A requested size of 3,000 differs from the realized sample count and from 3,000 training epochs.*

## 3. See the season as an image

The saved dimensions are `(sample, feature, season_week, place)`, with shape `(N, 1, 64, 64)`. The first 53 week positions and 51 location positions contain the season; the remaining rows and columns are padding. There is one incidence channel. State order comes from `SeasonAxis`, not alphabetical state names.

![One training season arranged as a week-by-location image](../assets/notebook-stories/02-training-image.png)

*Sample 3000 from the archived `30S70M` dataset. Cyan lines delimit padding. A square root makes the display readable; the model's full normalization is applied later by the training loader.*

## 4. Return to epidemic curves for a sanity check

![Five archived training seasons plotted on the US grid](../assets/notebook-stories/02-training-us-grid.png)

*The notebook's final `plot_us_grid` call, using samples 3000–3004 from the saved `30S70M` file. The geographic arrangement makes state trajectories readable; the model itself sees the fixed location-column order above.*

Inspect both representations before training: the image should have the expected populated region and padding, and the state curves should have sensible shapes and magnitudes after scaling and gap filling. Compare the realized mixture sizes and source counts with your intended recipes.

Each saved NetCDF stores coordinates, `main_origins`, and `mix_cfg` alongside the values. Carry these files into step 3 and update `dataset_library()` to point to them. Training derives its transformations from the chosen dataset, and forecasting uses that same dataset to reconstruct the transformations.

??? tip "Use saved results from the Zenodo archive"

    Use the exact paper inputs in `influpaint-paper/influpaint_paper_reproduction_data/datasets/training/`:

    ```bash
    mkdir -p training_datasets
    cp influpaint-paper/influpaint_paper_reproduction_data/datasets/training/TS_*_2025-07-17.nc training_datasets/
    ```

    The files are `TS_100S_2025-07-17.nc`, `TS_70S30M_2025-07-17.nc`, `TS_30S70M_2025-07-17.nc`, and `TS_100M_2025-07-17.nc`. The batch configuration selects these dates explicitly.

[Previous: 1. Gather source datasets](compiling-data-sources.md) · [Next: 3. Train candidate models](training.md)
