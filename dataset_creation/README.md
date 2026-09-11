# Dataset creation

The two Jupytext notebooks for preparing the state-level influenza training data:

| Notebook | Input | Output |
| --- | --- | --- |
| `1-gather_all_flu_datasets_ipynb.py` | Surveillance records and simulated trajectories | `Flusight/flu-datasets/all_datasets.parquet` |
| `2-build_training_flu_datasets_ipynb.py` | The combined Parquet table and mixing rules | Four training mixtures in `training_datasets/` |

Open the scripts as notebook cells and use the **research repository root** as the working directory. Data paths are relative to that root.

The walkthrough explains [gathering the sources](../docs/workflows/compiling-data-sources.md) and [creating training data](../docs/workflows/build-training-datasets.md), including the exact saved files for skipping either step. The first notebook stops deliberately before its optional custom-data section; the second writes date-stamped files and performs randomized mixing.

The paper uses the archived July 17, 2025 NetCDF datasets. Metrocast data preparation lives with its application under `satellite_applications/metrocast/`.
