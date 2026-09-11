# 0. Clone the repositories and install the environment

Start here before running the paper workflow. You will need Git and Conda. Run the commands below in a terminal; the remaining walkthrough assumes the `diffusion_torch` environment is active.

## Clone the research repository and CoPaint

```bash
git clone https://github.com/ACCIDDA/influpaint.git
cd influpaint
git clone https://github.com/jcblemai/CoPaint4influpaint.git CoPaint4influpaint
```

Clone **ACCIDDA/influpaint** to run the model. Keep `CoPaint4influpaint/` directly inside the research repository: the batch code imports its `guided_diffusion.O_DDIMSampler` from that location.

All subsequent commands run from the **research repository root**, the directory containing `influpaint/`, `main_training/`, and `mkdocs.yml`. The cloned dependencies and local data directories are ignored by Git and are not included in a fresh checkout.

## Create the Conda environment

On UNC Longleaf, connect to the login node using your UNC username and initialize Conda once:

```bash
ssh YOUR_ONYEN@longleaf.unc.edu
module purge
module load anaconda
conda init bash
```

Reconnect after initialization. Longleaf's shared base environment cannot be modified; create your own environment below. On other machines, start directly with these commands:

```bash
conda create -c conda-forge -n diffusion_torch \
  seaborn scipy numpy pandas matplotlib ipykernel \
  xarray netcdf4 h5netcdf tqdm einops tenacity \
  aiohttp ipywidgets jupyterlab pyarrow click jupytext pip

conda activate diffusion_torch

conda install pytorch torchvision -c pytorch
conda install -c bioconda epiweeks
python -m pip install mlflow pyyaml joblib blobfile

python -m ipykernel install --user --name diffusion_torch --display-name "Python (diffusion_torch)"
```

## Paper workflow dependencies

The environment above includes the Python dependencies used by training, inpainting, data preparation, and figure generation. GPU training and inpainting require a PyTorch installation compatible with the compute node's CUDA drivers. Check GPU access on the allocated GPU node:

```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Scoring also requires R with `scoringutils` and `dplyr`. In R, run:

```r
install.packages(c("scoringutils", "dplyr"))
```

The repository does not provide a pinned environment lockfile. The paper's Slurm launchers contain the original author's environment path; replace it with your environment's Python path and adapt the requested partition before submitting jobs. See [running notebooks and jobs on Longleaf](cluster.md).

## Use the reproduction archive

To reproduce saved results, place the extracted Zenodo reproduction archive at `influpaint-paper/influpaint_paper_reproduction_data/` under the research repository. Create the parent directory locally before extracting the archive.

The archive's `README.md` identifies the files for skipping each stage. See the [quick start](quick-start.md) for the figure-generation command and the [workflow overview](../workflows/start-here.md) for saved inputs at each stage.

## Build the documentation

```bash
python -m pip install mkdocs-material 'mkdocstrings[python]'
mkdocs build
```

To regenerate the captioned notebook stories from the reproduction archive's saved data:

```bash
python -m main_training.render_notebook_stories
```

## Source repositories

The clones below support source gathering and historical experiments. To follow the paper without downloading sources, use the archived checkpoints documented in the [workflow overview](../workflows/start-here.md).

```bash
git clone https://github.com/andreas128/RePaint.git archives/referenceimplementations/RePaint
git clone https://github.com/openai/guided-diffusion.git archives/referenceimplementations/guided-diffusion

git clone https://github.com/cdcepi/Flusight-forecast-data.git Flusight/2022-2023/FluSight-forecast-hub-official
git clone https://github.com/cdcepi/FluSight-forecast-hub Flusight/2023-2024/FluSight-forecast-hub-official
git clone https://github.com/cdcepi/FluSight-forecast-hub Flusight/2024-2025/FluSight-forecast-hub-official
git clone https://github.com/midas-network/flu-scenario-modeling-hub.git Flusight/Flu-SMH
git clone https://github.com/ACCIDDA/NC_Forecasting_Collab.git custom_datasets/NC_Forecasting_Collab
git clone https://github.com/reichlab/flu-metrocast.git Flusight/metrocast/flu-metrocast

# Historical WIS implementation by Adrian Lison
git clone https://github.com/adrian-lison/interval-scoring.git interval_scoring
```

## Update data

The Delphi API client is included in the repository at `helpers/delphi_epidata.py`; no separate Delphi clone is required.

After cloning the data repositories above, run from the research repository root. This pulls the local FluSight, NC collaboration, and Metrocast checkouts.

```bash
./update-data.sh
```

## Git LFS on Longleaf

For repositories that store data with Git LFS, install it in your own environment and fetch their large files from inside the relevant checkout:

```bash
conda install -c conda-forge git-lfs
git lfs install
git lfs pull
```

## Open the notebooks

Launch `jupyter lab` from the research repository root and select **Python (diffusion_torch)** as the notebook kernel. The current data notebooks are in `dataset_creation/`; [step 1](../workflows/compiling-data-sources.md) explains opening the Jupytext source as a notebook. Forecast generation is covered in [step 5](../workflows/inpainting.md). Earlier instructions referenced `dataset_builder.ipynb` and `inpaintingFluForecasts.ipynb`; use the current walkthrough for the paper workflow.

For remote notebooks, follow [the Longleaf instructions](cluster.md).

[Next: Workflow overview](../workflows/start-here.md)
