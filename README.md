# InfluPaint: Generative diffusion models for spatiotemporal influenza forecasting

* **Authors:** Joseph Lemaitre, Justin Lessler
* **Affiliation:** The University of North Carolina at Chapel Hill

InfluPaint represents influenza seasons as images, with weeks and locations as axes and incidence as pixel intensity. A denoising diffusion model learns from surveillance and simulated trajectories; CoPaint conditions generated seasons on observed data to forecast future hospitalizations or reconstruct missing observations without retraining.

[Read the paper on arXiv](https://arxiv.org/abs/2604.24913) · [Documentation](docs/index.md)

## Running InfluPaint

Start with [installing the environment and cloning the repositories](docs/getting-started/installation.md), the first page of the walkthrough. Follow the [workflow guide](docs/workflows/start-here.md) for source data, training, inpainting, evaluation, and paper figures, or use the [quick start](docs/getting-started/quick-start.md) to reproduce figures from saved results. [Cluster and notebook instructions](docs/getting-started/cluster.md) cover UNC Longleaf.

### Paper reproducibility

To plot the figures in the paper, follow [Reproduce the paper figures](docs/workflows/paper-figures.md). This uses the saved data and results in the Zenodo reproducibility archive. **Zenodo DOI: [10.5281/zenodo.22699980](https://doi.org/10.5281/zenodo.22699980).**

See the [Zenodo archive README](docs/reproducibility/README.md) for archive contents and provenance.

## Forecast influenza hospitalizations

[![Paper Figure 2: four-week influenza hospitalization forecasts for two seasons](docs/assets/paper/figure-2-forecasts.png)](docs/assets/paper/figure-2-forecasts.png)

**Figure 2 — Forecasts from observed history.** InfluPaint conditions on the observed part of a season to generate 512 possible trajectories for future hospitalizations. Colored fans show forecast uncertainty and colored lines show medians for North Carolina, New York, Texas, and Florida across the 2023–2024 and 2024–2025 seasons. Black lines show observed hospitalizations, dotted lines show the FluSight ensemble, and dashed vertical lines mark the last observed week for each forecast. These retrospective forecasts use finalized observations up to each forecast date.

## Adapt to different observation patterns without retraining

[![Paper Figure 4: reconstructions with missing states, missing weeks, and checkerboard observation masks](docs/assets/paper/figure-4-reconstruction.png)](docs/assets/paper/figure-4-reconstruction.png)

**Figure 4 — One trained model, many reconstruction tasks.** InfluPaint can adapt to different patterns of missing data by changing the observation mask at sampling time, without retraining the model. The same model reconstructs missing states, fills midseason gaps, infers early-season dynamics from later observations, and handles checkerboard patterns of missing weeks and locations. Black curves show observed hospitalizations; colored fans and lines show predictive quantiles and medians. The insets identify observed entries in green and hidden entries in red. This flexibility lets the model work with partial spatial coverage and interrupted time series.

## How InfluPaint works

[![Paper Figure 5: encoding epidemic seasons as images, learning to denoise, and conditioning generation with an observation mask](docs/assets/paper/figure-5-methods.png)](docs/assets/paper/figure-5-methods.png)

**Figure 5 — Model overview.** **a.** An influenza season becomes an image whose axes represent weeks and locations and whose pixel intensity represents incidence. **b.** A diffusion model learns to reverse the gradual addition of noise, allowing it to generate new, plausible seasons. **c.** Inpainting combines observed values and a mask with the generation process to infer the missing parts of a season. The paper's forecasting implementation uses CoPaint to condition these generated trajectories on the available observations.

## Repository map

| Directory | Purpose |
| --- | --- |
| [`influpaint/`](influpaint/) | Shared models, datasets, batch code, and utilities |
| [`dataset_creation/`](dataset_creation/README.md) | The two main source-gathering and training-data notebooks |
| [`evaluation/`](evaluation/README.md) | Forecast scoring, model comparison, and analysis exports |
| [`main_training/`](main_training/README.md) | Paper training, calibration, and batch launchers |
| [`paper_figures/`](paper_figures/README.md) | Paper figure generation |
| [`docs/`](docs/index.md) | Paper overview and step-by-step walkthrough |
| [`satellite_applications/`](satellite_applications/README.md) | Metrocast and RSV applications |
| [`archives/`](archives/README.md) | Development notebooks, historical workflows, and reference implementations |
| `CoPaint4influpaint/` | Active CoPaint runtime dependency |
| `Flusight/`, `training_datasets/`, `from_longleaf/`, `model_candidate_evaluation/` | Local data and results |

## Research process and early development

These notes preserve the original model-development process and early FluSight experience. The RePaint experiments, architecture settings, timing, and research priorities below are historical; the current paper workflow uses CoPaint, as described in the [architecture documentation](docs/architecture/inpainting.md).

### Introduction

Denoising Diffusion Probabilistic Models (DDPM) are generative models like generative adversarial networks, autoregressive models, and variational auto-encoder. While slow, they can generate high quality samples with good diversity. They work wonderfully well for image synthesis, see [openAI DALL-E 2](https://openai.com/dall-e-2/). These model has been described in [1] and [2].

Here we treat influenza epidemic curves as images (axes are time and location, a pixel value is e.g incident hospitalization), and we generate synthetic epidemic trajectories using “stock” DDPM algorithms.

### Denoising Diffusion Probabilistic Models (DDPM)
DDPM consists of two transforms, backward and forward, in diffusion time:
* Forward: Markov chain that gradually adds noise to the data until the signal is destroyed;
* Backward: A trained neural network that denoises the image step by step.

We train the neural network using forward transform samples from the dataset, and we sample by transforming random gaussian noise with the backward transform.

<img width="1345" alt="image" src="https://user-images.githubusercontent.com/7485811/233663274-fb44d45c-52f2-4b3f-82f2-5b596c180e99.png">

### Neural Net Architecture

_(note that this section is updated less often than the code and might be outdated as we frequently improve the model)_

This architecture was built from trial & error (e.g, ConvNext blocks yield unsatisfying results) and is empirically satisfying. We use as a Forward process (or noise scheduler) a linear beta-schedule with 200 time-steps. The neural network for the backward transform is a  U-net with a Wide ResNet block, Attention module, Group normalization, and Residual connection. 

We use a Sinusoidal time-step embedding to integrate time-step information into the sample (as the neural network weights are the same for any diffusion time).

This architecture is directly taken from image generation DDPM. Treating forecasts as images has many caveats as the locality in the incidence and time axes have very different semantics (e.g human mobility between states is not properly captured by the small convolution kernels). We're exploring custom architecture but found that the present one performs empirically well. The code is directly inspired from PyTorch implementations found, especially the [Hugging Face Annotated Diffusion model](https://huggingface.co/blog/annotated-diffusion), but also [Keras docs](https://keras.io/examples/generative/ddim/) and this [Colab notebook](https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=Rj17psVw7Shg).

### Inpainting
With our DDPM, we can generate epidemiological curves of Flu Hospitalization, i.e full season forecasts. However, after the first data point are reported, we condition our forecasts on the evidence using inpainting. Inpainting alter the reverse diffusion iterations by sampling the unmasked regions using ground-truth. We use the REpaint algorithm [3], which outperform SOTA (GAN, Auto-regressive) for diverse, high-quality inpainting. 

<img width="670" alt="image" src="https://user-images.githubusercontent.com/7485811/219053390-b0aecee5-db2d-431e-8a7b-3a771f18d396.png">

The Repaint algorithm has important parameters that influence global harmonization: the number of resampling steps and possible jumps in diffusion time. We use repaint for forecasts with past incidence as ground truth:

<img width="725" alt="image" src="https://user-images.githubusercontent.com/7485811/219055262-69e3c059-176e-4311-9aa7-8a98846933ea.png">

In Feb 2023, Rout et al. showed that the RePaint algorithm has some theoretical justification for sample recovery. But they also observed and corrected a misalignment error between the inpainted areas and the ground-truth. We have been observing this as well (see image below) and we have been working hard to correct it [4]

### Training Data
No dataset corresponds to what we want to model (New Hospital Admission at the state level). We use either data generated by a mechanistic influenza transmission model (taken from Round 1 of the [US Flu Scenario Modeling Hub](https://fluscenariomodelinghub.org)) or reported US influenza data ([FluView](https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html), [FluSurv](https://www.cdc.gov/flu/weekly/influenza-hospitalization-surveillance.htm)) at different locations, or more generally a mix of the above in certain proportions. We augment the dataset with random transform.

### FluSight
We submit weekly Flu hospitalization forecasts for each U.S. state to the FluSight challenge, organized by the CDC. We have found that InfluPaint is been plug-n-play, but good performance required a bit of “care”: mainly applying the right transforms to enrich the dataset, having the right diffusion timing and the right inpainting parameters, such as resampling and jumps. It is not too computationally intensive (only 30 min on a Tesla V100-16GB, training included). 

A benefit of this approach is the nice diversity in forecasts compared to mechanistic models (double peaks and single peaks, little bumps …), but our performance is inequal as we iteratively improved the algorithm for this Flu season.

<img width="988" alt="image" src="https://user-images.githubusercontent.com/7485811/233660673-b415dd62-fd5a-4097-b8ce-ef698202269d.png">


### Research directions from the early experiments
- Use other variables (humidity, Flu A, and Flu B proportions) as additional image channels.
- Application to other diseases and validation
- Design of ID modeling specific neural architectures

[1] Ho, Jonathan, Ajay Jain, and Pieter Abbeel. “Denoising Diffusion Probabilistic Models.” arXiv, December 16, 2020. https://doi.org/10.48550/arXiv.2006.11239.

[2] Dhariwal, Prafulla, and Alex Nichol. “Diffusion Models Beat GANs on Image Synthesis.” arXiv, June 1, 2021. https://doi.org/10.48550/arXiv.2105.05233.

[3] Lugmayr, Andreas, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. “RePaint: Inpainting Using Denoising Diffusion Probabilistic Models.” arXiv, August 31, 2022. https://doi.org/10.48550/arXiv.2201.09865.

[4] Rout, Parulekar, aramanis, Shakkottai. “A Theoretical Justification for Image Inpainting Using Denoising Diffusion Probabilistic Models.” arXiv, February 2, 2023. https://arxiv.org/abs/2302.01217

## Log

2023-04-11 Our model showed some bias in the CDC's plots that we did not observe on our side, and we have found a bug where the post-processing rescaling (due to the misalignement observed in Rout et al.) has not been applied in the quantiles sent to FluSight. This mainy affected the submissions of the last few weeks, where our model consistently scaled lower than reported hospital admission. It is now corrected.

2023-02-13: Before: more resampling steps gives smaller confidence interval, because more certainty in what matches the curve. Now that I added some random noise perturbation of the training set, more resampling steps gives larger confidence interval under certain conditions.
