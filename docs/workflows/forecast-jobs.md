# 4. Prepare forecast jobs

Give each trained candidate the same forecast dates and conditioning settings so you can compare their predictions. Starting from the finished training runs in step 3, choose the reference dates and conditioning settings, then use `influpaint.batch.generate_inpainting_jobs` to write a job manifest and a Slurm launcher.

A **forecast job** is one trained checkpoint, one forecast reference date, and one CoPaint configuration. A **manifest** is the table listing those jobs. A **Slurm array** submits that table as multiple cluster tasks, with each task running one row. It will generate an ensemble of possible seasons conditioned on the history available under that date's observation mask. This step defines the experiment; step 5 runs the sampling.

!!! tip "Paper reproducibility"

    Get the results for this step directly from the [Zenodo reproducibility archive](start-here.md#reproducibility-archive).

    The paper manifest is `main_training/paper_runs_2025-07-22/inpaint_jobs_paper-2025-07-22.txt`, with its launcher at `main_training/inpaint_array_paper-2025-07-22.run`. These refer to the historical MLflow runs. To explore saved selected-model forecasts directly, use `forecasts/retrospective/` in the Zenodo archive; the [Zenodo archive README](../reproducibility/README.md) describes the coverage.

## 1. Select the training runs

The generator's `-e` argument names an MLflow training experiment. It queries the active tracking store for runs with status `FINISHED`, reads each run's scenario ID and run ID, and sorts them by scenario ID. It includes every finished run returned by that query, so choose an experiment containing the candidates you intend to compare.

The scenario ID reconstructs the architecture, dataset, and transformations. The run ID selects the learned checkpoint. If you train the same scenario again, the new run has its own weights and run ID.

## 2. Define the forecast calendar

Edit the `flu_seasons` list in `influpaint/batch/generate_inpainting_jobs.py` to choose each season's date range and frequency. The generator uses `pandas.date_range` to expand these settings. The supplied comparison uses every other Saturday (`2W-SAT`):

| Season | Configured start / end | Reference dates produced |
| --- | --- | --- |
| 2023–2024 | `2023-10-14` / `2024-05-04` | 15 dates, October 14 through April 27 |
| 2024–2025 | `2024-11-23` / `2025-05-31` | 14 dates, November 23 through May 24 |

The final reference date can precede the configured end because of the two-week spacing. For a different experiment, edit this list before generating jobs; date ranges are not command-line options. These reference dates determine the historical observation masks used in step 5.

## 3. Choose how to condition the model

CoPaint settings control how the sampler fits the observations while generating the unobserved part of a season. They are defined by `copaint_config_library()` in `influpaint/batch/config.py` and selected with `--configs`:

| Configuration | Time travel | Jump length | Latent optimization iterations |
| --- | --- | ---: | ---: |
| `celebahq_noTTJ5` | Disabled | 5 (inactive) | 2 |
| `celebahq_try3` | Enabled | 5 | 5 |
| `celebahq` | Enabled | 10 | 2 |

A latent optimization iteration adjusts the current noisy image to reduce disagreement with the observed entries. The iteration count is the number of these adjustments in an optimization call during sampling. Jump length is measured in diffusion steps, not epidemic weeks. Time travel revisits noisier states during sampling, giving the trajectory further opportunities to reconcile with observations. These settings change sampling while keeping the trained network weights fixed. Omitting `--configs` uses `AVAILABLE_COPAINT_CONFIGS`, which currently contains the three settings above.

## 4. Generate the manifest and launcher

Use the same experiment name you chose in step 3:

```bash
python -m influpaint.batch.generate_inpainting_jobs \
  -e my-flu-experiment_training \
  --configs celebahq_noTTJ5,celebahq_try3,celebahq \
  --output_dir main_training/generated
```

The generator takes the Cartesian product of **finished training runs × reference dates × conditioning configurations**. For example, 15 runs across the supplied 29 dates and three configurations produce **1,305 jobs**. Your count depends on the finished runs in your experiment.

It writes two files under `main_training/generated/`:

| File | Purpose |
| --- | --- |
| `inpaint_jobs_my-flu-experiment.txt` | A comma-separated manifest describing each forecast job |
| `inpaint_array_my-flu-experiment.run` | A Slurm array launcher that reads one manifest row per task |

Each manifest row has these fields:

```text
job_id,scenario_id,run_id,season,date,config
```

Here is the first row of the saved paper manifest:

```csv
job_id,scenario_id,run_id,season,date,config
0,36,ad86a8ef3da4442381cbab309cea952b,2023-2024,2023-10-14,celebahq_noTTJ5
```

| Field | Meaning in this row |
| --- | --- |
| `job_id = 0` | The first generated job; Slurm array task 0 reads the first data row after the header |
| `scenario_id = 36` | Use the 200-step linear diffusion candidate from the training comparison |
| `run_id` | Load the checkpoint from this specific MLflow training execution |
| `season = 2023-2024` | Label this job as part of the 2023–2024 comparison |
| `date = 2023-10-14` | Make the forecast for this reference date; observations before its mask boundary are supplied to the model |
| `config = celebahq_noTTJ5` | Condition with time travel disabled and two latent optimization iterations |

The next two rows use the same training run and date with `celebahq_try3` and `celebahq`. After that, the generator advances to the next date. Each training run has 29 × 3 = 87 jobs in this comparison.

The launcher selects rows by their physical position in the file, skipping the header. It passes the scenario, run ID, date, and configuration to `influpaint.batch.inpainting`; that script derives the season from the date. The `season` field is also used in the task's log filename. Keep one header and one job per line when inspecting or editing a manifest.

## 5. Review the experiment before submitting

Open the manifest and check that it contains the intended candidates, dates, and settings. Then edit the generated `.run` file for your Python environment and cluster resources. Its template requests one GPU, 32 GB RAM, and four hours per task on the original Longleaf partitions. It names the forecast experiment `my-flu-experiment_inpainting`.

For repeated batches, `--skip_completed` removes combinations already marked `FINISHED` in the matching inpainting experiment. Completion is matched by scenario, date, and configuration, without the training run ID. Use a new experiment name when comparing newly trained weights. The result-directory names also omit the run ID, so keep one run per scenario in an experiment to avoid output collisions.

Submit this launcher in [step 5](inpainting.md). Each task writes a log named `out_inpaint_s<scenario>_<season>_<date>_<config>.out`; use that name to find the log for a particular manifest row.

[Previous: 3. Train candidate models](training.md) · [Next: 5. Generate forecasts](inpainting.md)
