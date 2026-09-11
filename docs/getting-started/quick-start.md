# Quick start

To understand the full process, follow [Walkthrough → Start here](../workflows/start-here.md). To recreate the paper figures, place the reproduction archive at `influpaint-paper/influpaint_paper_reproduction_data/` and run from the research repository root:

```bash
python -m paper_figures.final_figures --data-root influpaint-paper/influpaint_paper_reproduction_data
```

Outputs go to the archive's `regenerated_paper_figures/`. See [step 8](../workflows/paper-figures.md) for the figure/input map and [Installation](installation.md) for the environment.

For a fresh calibration, restore the exact training datasets as shown in [step 2](../workflows/build-training-datasets.md), then follow steps 3–7. `main_training/runs.txt` collects the commands; its Slurm launchers must be submitted from the repository root. New jobs are generated under `main_training/generated/` after training finishes.
