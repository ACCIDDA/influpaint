# Quick start

To prepare data, train models, and make forecasts, follow [the walkthrough](../workflows/start-here.md).

For reproducibility, download the Zenodo archive (DOI: [10.5281/zenodo.22699980](https://doi.org/10.5281/zenodo.22699980)). To plot the figures in the paper, place the extracted archive at `influpaint-paper/influpaint_paper_reproduction_data/` and run from the research repository root:

```bash
python -m paper_figures.final_figures --data-root influpaint-paper/influpaint_paper_reproduction_data
```

Outputs go to the archive's `regenerated_paper_figures/`. See [Reproduce the paper figures](../workflows/paper-figures.md) for the output filenames and the saved data used by each plot. Install the environment using [these instructions](installation.md).

For a fresh calibration, restore the exact training datasets as shown in [step 2](../workflows/build-training-datasets.md), then follow steps 3–7. `main_training/runs.txt` collects the commands; its Slurm launchers must be submitted from the repository root. New jobs are generated under `main_training/generated/` after training finishes.
