# 3. Train candidate models

Train diffusion models on the season images from step 2, then compare how their architecture, data mixture, and preprocessing affect the seasons they generate. Each **candidate model** combines a dataset mixture, an image transformation, optional data augmentation, a U-Net architecture, and a diffusion process. Training produces a checkpoint for each candidate; later steps evaluate how well those checkpoints forecast observed epidemics.

Use `influpaint/batch/config.py` to define these choices, `influpaint/batch/scenarios.py` to combine them, and `influpaint.batch.training` to train one candidate at a time.

!!! tip "Paper reproducibility"

    You can bypass training the paper's selected model by getting its final output, `i868::m_U500cRx1224::ds_30S70M::tr_Sqrt::ri_No::3000.pth`, directly from the [Zenodo reproducibility archive](https://doi.org/10.5281/zenodo.22699980) ([archive README](../reproducibility/README.md)). Training-loss tables for the candidate models are also saved under `analysis/`.

    If you have not completed step 2, you can start training with the four `TS_*_2025-07-17.nc` files in `datasets/training/` from the archive. Copy them into `training_datasets/` as shown in [step 2](build-training-datasets.md).

    The generated batch workflow uses MLflow run IDs; to load a standalone checkpoint directly, use the `-m` option described in step 5 or the operational notebook in step 9.

## 1. Understand what defines a candidate

The five option lists in `config.py` form the training search space:

| Choice | Available settings | What changes |
| --- | --- | --- |
| Diffusion process | `U200l`, `U200c`, `U500l`, `U500c`, `U800c` | Number of diffusion steps (200, 500, or 800) and a linear (`l`) or cosine (`c`) noise schedule |
| U-Net | `Rx124`, `Rx1224`, `Cx1224`, `Rx12448` | Residual (`R`) or ConvNeXt (`C`) blocks and the channel multipliers at successive network levels; the base width is 64 |
| Training mixture | `100S`, `70S30M`, `30S70M`, `100M` | Surveillance only; 70% surveillance / 30% simulations; 30% surveillance / 70% simulations; or simulations only. Surveillance comprises FluView and FluSurv; simulations comprise flepiMoP and Flu Scenario Modeling Hub trajectories. |
| Transformation | `Lins`, `Sqrt`, `LinsZs`, `LogZs` | Linear scaling, square-root scaling, or the linear/log standardization recipes implemented in `transform_library` |
| Enrichment | `No`, `Pois`, `PoisPadScaleSmall`, `PoisPadScale` | No augmentation, Poisson resampling, or Poisson resampling combined with random time shifts and magnitude scaling |

### What the model learns

A **diffusion step** is one level of noise in the process that corrupts a season image. Training draws a noise level, corrupts a training image at that level, and asks the network to predict the added noise. A **noise schedule** determines how much corruption each level represents. When generating a season, the model starts from noise and reverses this process.

The **U-Net** is the neural network that predicts the noise. It reduces the image to coarser representations and then expands it back to the original resolution, with connections between corresponding levels. The architecture codes specify its blocks and feature widths. For example, `Rx1224` uses residual blocks and multipliers `(1, 2, 2, 4)` on a base width of 64, giving feature widths of 64, 128, 128, and 256 at those levels. `Cx1224` uses the same multipliers with ConvNeXt blocks.

### What preprocessing and enrichment do

A **transformation** changes the numerical scale seen by the model and has an inverse used to return generated values to the hospitalization scale. With square-root preprocessing (`Sqrt`), let `M` be the largest value in the training dataset's incidence channel. An input value `x` becomes `2 × sqrt(x / M)`, and a generated value `y` becomes `M × (y / 2)²` after inversion. For example, if `M = 10,000`, an input count of 100 becomes 0.2. This reduces the dominance of large peaks in the training values.

`Lins` uses `2 × x / M`. `LinsZs` subtracts the dataset mean and divides by its standard deviation. `LogZs` uses the code's logarithmic recipe: `(log(1 + x) - log(mean)) / log(std)`, where the mean and standard deviation come from the untransformed dataset. The code recomputes these statistics when loading a model, which is why forecasting needs the same training dataset as well as the checkpoint.

**Enrichment** changes a training example randomly each time it is loaded. Poisson resampling replaces each value with a draw whose mean is that value, adding count variation. Time shifts move an epidemic earlier or later and fill the exposed weeks with zeros. Magnitude scaling multiplies the whole image by a random factor. These changes happen before the numerical transformation; they are not inverted when generating forecasts.

Enrichment is applied when a training image is loaded, before the transformation. It adds variation to the saved dataset: the small recipe shifts by up to four weeks and scales by 0.7–1.3; the wider recipe uses up to 15 weeks and 0.1–1.9. The dataset mixture itself is chosen in step 2.

`get_all_training_scenarios()` enumerates the Cartesian product of the five lists: **5 × 4 × 4 × 4 × 4 = 1,280 possible candidates**. A scenario ID is the zero-based position in that enumeration. Its readable name records the choices, for example:

```text
i868::m_U500cRx1224::ds_30S70M::tr_Sqrt::ri_No
```

This means 500 diffusion steps with a cosine schedule, a residual U-Net with channel multipliers `(1, 2, 2, 4)`, the 30% surveillance / 70% modeling mixture, a square-root transformation, and no enrichment. Changing option-list order changes the numeric IDs, so use the readable formulation to check your selection.

## 2. Choose the comparison you want to run

The paper uses a smaller comparison built around `CONFIG_BASELINE`: **i804**, with `U500c`, `Rx124`, `30S70M`, `Sqrt`, and `No`. `get_essential_scenarios()` selects this baseline plus every candidate that changes exactly one of those five choices. That gives **17 requested candidates**, making it possible to examine the effect of each choice while holding the others fixed. For example, i868 changes only the U-Net from `Rx124` to `Rx1224`.

The supplied Slurm array requests the following comparisons. Every row keeps all baseline settings except the named change:

| Comparison | Scenario IDs and changed settings |
| --- | --- |
| Baseline | **804:** 500 cosine diffusion steps, `Rx124`, 30% surveillance / 70% simulations, square root, no enrichment |
| Diffusion process | **36:** 200 linear; **292:** 200 cosine; **548:** 500 linear; **1060:** 800 cosine |
| Network | **868:** `Rx1224`; **932:** `Cx1224`; **996:** `Rx12448` |
| Training data | **772:** surveillance only; **788:** 70% surveillance / 30% simulations; **820:** simulations only |
| Transformation | **800:** linear scaling; **808:** linear standardization; **812:** logarithmic recipe |
| Enrichment | **805:** Poisson, wide shifts and scaling; **806:** Poisson, small shifts and scaling; **807:** Poisson only |

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

### What an epoch and a batch mean

A **batch** is a group of training images processed in one optimizer update. An **epoch** is one pass through the shuffled training loader. With 3,223 images and a batch size of 512, the loader produces six full batches per epoch and drops the remaining 151 images for that pass. Shuffling changes which images are dropped in later epochs. At 3,000 epochs, that gives 18,000 optimizer updates. Diffusion steps, training epochs, and optimizer updates count different operations.

## 4. Follow what the training script does

For each candidate, the script:

1. Resolves the scenario, loads its NetCDF dataset, and computes dataset-dependent scaling, mean, and standard deviation. It attaches the selected enrichment, transformation, and inverse transformation to the loader.
2. Builds the U-Net and diffusion schedule, then creates shuffled batches of season images. The loader drops incomplete batches, so the dataset must contain at least one full batch.
3. Chooses a random diffusion time for each image, adds Gaussian noise, and trains the U-Net to predict that noise. The configured models minimize mean squared error using Adam. Repeating this over the epochs teaches the network to reverse the corruption process.
4. Saves the trained checkpoint and logs it to MLflow together with the scenario, dataset statistics, output paths, and loss metrics.
5. Starts from noise to generate unconditional seasons, inverse-transforms them to the original scale, and saves sample arrays and diagnostic plots.

These unconditional samples show what the model has learned before any current-season observations are supplied. Forecast conditioning happens in step 5.

### Files and records created by a training run

A **checkpoint** is a `.pth` file containing the network weights, optimizer state, epoch setting, and loss type. For the i868 example, its filename ends with:

```text
i868::m_U500cRx1224::ds_30S70M::tr_Sqrt::ri_No::3000.pth
```

The script creates a directory under `training_output/` named with the code revision, experiment name, and run date. It records that directory as `output_folder` and the checkpoint as `model_path` in MLflow.

**MLflow** stores experiment records. An experiment groups related training runs; a run records one training execution. The script logs `step_loss`, `epoch_loss`, the final loss and the average of the last 100 training losses. It also saves the checkpoint under the run's `checkpoints` artifacts and generated arrays under `samples`. Sample plots show the generated epidemic curves after inverse transformation.

## 5. Inspect the result before forecasting

Look at the loss curves for convergence or divergence, then inspect the generated epidemic curves for plausible timing, magnitude, and variation across locations. A low denoising loss alone does not establish forecasting skill; the retrospective forecasts and scores in steps 4–6 provide that comparison.

Carry forward the **MLflow experiment name**, each finished **run ID**, and the corresponding training datasets. A scenario ID describes a recipe; a run ID identifies one execution and its learned weights. The forecast generator queries finished runs and expands each one into date-specific jobs.

[Previous: 2. Create training data](build-training-datasets.md) · [Next: 4. Prepare forecast jobs](forecast-jobs.md)
