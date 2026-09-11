# Inpainting architecture

CoPaint conditions the learned season distribution on observed entries. A mask identifies which weeks and locations are observed. The sampler adjusts reverse diffusion using this evidence while generating plausible values in the hidden region. Hiding future weeks creates forecasts; other masks reconstruct missing locations or time intervals.

The paper batch runner in `influpaint/batch/inpainting.py` uses `CoPaint4influpaint`'s `O_DDIMSampler`. Its selected configuration, `celebahq_noTTJ5`, disables time travel and uses two latent optimization iterations. `influpaint/models/inpaint_module.py` contains older inpainting code and is not the sampler used in this batch workflow.

Each forecast job loads the trained model using the manifest's MLflow run ID, builds its observation mask, samples 512 trajectories, inverse-transforms them to hospitalization counts, and exports marginal quantiles. See [steps 4–5](../workflows/forecast-jobs.md) for job generation and submission, and [step 7](../workflows/mask-experiments.md) for reconstruction experiments.
