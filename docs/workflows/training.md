# 3. Train candidate models

The objective is to learn a distribution of complete influenza seasons from the training images created in step 2, and to compare how modeling choices affect that distribution. Each **candidate model** combines a dataset mixture, an image transformation, optional data augmentation, a U-Net architecture, and a diffusion process. Training produces a checkpoint for each candidate; later steps evaluate how well those checkpoints forecast observed epidemics.

Use `influpaint/batch/config.py` to define these choices, `influpaint/batch/scenarios.py` to combine them, and `influpaint.batch.training` to train one candidate at a time.

## 1. Understand what defines a candidate

The five option lists in `config.py` form the training search space:

| Choice | Available settings | What changes |
| --- | --- | --- |
| Diffusion process | `U200l`, `U200c`, `U500l`, `U500c`, `U800c` | Number of diffusion steps (200, 500, or 800) and a linear (`l`) or cosine (`c`) noise schedule |
| U-Net | `Rx124`, `Rx1224`, `Cx1224`, `Rx12448` | Residual (`R`) or ConvNeXt (`C`) blocks and the channel multipliers at successive network levels; the base width is 64 |
| Training mixture | `100S`, `70S30M`, `30S70M`, `100M` | How much surveillance (`S`) and modeled (`M`) data the network sees |
| Transformation | `Lins`, `Sqrt`, `LinsZs`, `LogZs` | Linear scaling, square-root scaling, or the linear/log standardization recipes implemented in `transform_library` |
| Enrichment | `No`, `Pois`, `PoisPadScaleSmall`, `PoisPadScale` | No augmentation, Poisson resampling, or Poisson resampling combined with random time shifts and magnitude scaling |

Enrichment is applied when a training image is loaded, before the transformation. It adds variation to the saved dataset: the small recipe shifts by up to four weeks and scales by 0.7–1.3; the wider recipe uses up to 15 weeks and 0.1–1.9. The dataset mixture itself is chosen in step 2.

`get_all_training_scenarios()` enumerates the Cartesian product of the five lists: **5 × 4 × 4 × 4 × 4 = 1,280 possible candidates**. A scenario ID is the zero-based position in that enumeration. Its readable name records the choices, for example:

```text
i868::m_U500cRx1224::ds_30S70M::tr_Sqrt::ri_No
```

This means 500 diffusion steps with a cosine schedule, a residual U-Net with channel multipliers `(1, 2, 2, 4)`, the 30% surveillance / 70% modeling mixture, a square-root transformation, and no enrichment. Changing option-list order changes the numeric IDs, so use the readable formulation to check your selection.

## 2. Choose the comparison you want to run

The paper uses a smaller comparison built around `CONFIG_BASELINE`: **i804**, with `U500c`, `Rx124`, `30S70M`, `Sqrt`, and `No`. `get_essential_scenarios()` selects this baseline plus every candidate that changes exactly one of those five choices. That gives **17 requested candidates**, making it possible to examine the effect of each choice while holding the others fixed. For example, i868 changes only the U-Net from `Rx124` to `Rx1224`.

Print the available formulations and the essential Slurm array list:

```bash
python -m influpaint.batch.scenarios
```

For your own comparison, edit the libraries and `CONFIG_BASELINE`, inspect the printed scenario strings, and copy the desired IDs into `#SBATCH --array` in `main_training/train.run`. The launcher contains a fixed list; it does not regenerate the essential scenarios when submitted. Training epochs and batch size are separate command-line settings, with defaults of 3,000 and 512.

Before training, check `dataset_library()` in `config.py`. It currently points to the four `TS_*_2025-07-17.nc` files. If step 2 generated new datasets, update this mapping to those filenames and use the same mapping when forecasting with the resulting checkpoints.

## 3. Run one candidate or submit the array

On a machine with the training environment and GPU, one candidate can be trained directly:

```bash
python -m influpaint.batch.training \
  -s 868 -e my-flu-experiment_training \
  -d ./training_output/ --epochs 3000 --batch_size 512
```

The `-s` argument selects the scenario, `-e` names the MLflow experiment that groups your runs, and `-d` sets the checkpoint output root. Keep the trailing slash on that output path: the script appends its experiment-directory name directly. Use an experiment name ending in `_training` so the forecast-job generator can derive the matching `_inpainting` experiment.

For the full comparison on Slurm, edit `main_training/train.run` to set your experiment name, Python executable, partition, and scenario IDs. Set the output root in your environment before submitting:

```bash
export OCP_OUTDIR="$PWD/training_output/"
sbatch main_training/train.run
```

The supplied launcher requests one GPU, 32 GB RAM, and 12 hours per candidate. Each array task passes its `SLURM_ARRAY_TASK_ID` as the scenario ID to `influpaint.batch.training`. Keep the same MLflow tracking store available for the job-generation step; that store connects each training run to its checkpoint artifacts.

## 4. Follow what the training script does

For each candidate, the script:

1. Resolves the scenario, loads its NetCDF dataset, and computes dataset-dependent scaling, mean, and standard deviation. It attaches the selected enrichment, transformation, and inverse transformation to the loader.
2. Builds the U-Net and diffusion schedule, then creates shuffled batches of season images. The loader drops incomplete batches, so the dataset must contain at least one full batch.
3. Chooses a random diffusion time for each image, adds Gaussian noise, and trains the U-Net to predict that noise. The configured models minimize mean squared error using Adam. Repeating this over the epochs teaches the network to reverse the corruption process.
4. Saves the trained checkpoint and logs it to MLflow together with the scenario, dataset statistics, output paths, and loss metrics.
5. Starts from noise to generate unconditional seasons, inverse-transforms them to the original scale, and saves sample arrays and diagnostic plots.

These unconditional samples show what the model has learned before any current-season observations are supplied. Forecast conditioning happens in step 5.

## 5. Inspect the result before forecasting

Look at the loss curves for convergence or divergence, then inspect the generated epidemic curves for plausible timing, magnitude, and variation across locations. A low denoising loss alone does not establish forecasting skill; the retrospective forecasts and scores in steps 4–6 provide that comparison.

Carry forward the **MLflow experiment name**, each finished **run ID**, and the corresponding training datasets. A scenario ID describes a recipe; a run ID identifies one execution and its learned weights. The forecast generator queries finished runs and expands each one into date-specific jobs.

??? tip "Use saved results from the Zenodo archive"

    The archive contains the selected `i868::m_U500cRx1224::ds_30S70M::tr_Sqrt::ri_No::3000.pth` checkpoint and training-loss tables under `analysis/`. Its README documents the saved run and dataset. The generated batch workflow uses MLflow run IDs; to load a standalone checkpoint directly, use the `-m` option described in step 5 or the operational notebook in step 9.

[Previous: 2. Create training data](build-training-datasets.md) · [Next: 4. Prepare forecast jobs](forecast-jobs.md)
