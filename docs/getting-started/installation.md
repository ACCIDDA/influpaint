# Installation

## Conda Environment

Create the conda environment:

```bash
conda create -c conda-forge -n diffusion_torch \
  seaborn scipy numpy pandas matplotlib ipykernel \
  xarray netcdf4 h5netcdf tqdm einops tenacity \
  aiohttp ipywidgets jupyterlab

conda activate diffusion_torch

conda install torchvision -c pytorch
conda install -c bioconda epiweeks

python -m ipykernel install --user --name diffusion_torch --display-name "Python (diffusion_torch)"
```

## Clone Required Repositories

```bash
git clone https://github.com/andreas128/RePaint.git referenceimplementations/RePaint
git clone https://github.com/openai/guided-diffusion.git referenceimplementations/guided-diffusion
git clone https://github.com/cmu-delphi/delphi-epidata.git Flusight/flu-datasets/delphi-epidata

git clone https://github.com/cdcepi/Flusight-forecast-data.git Flusight/2022-2023/FluSight-forecast-hub-official
git clone https://github.com/cdcepi/FluSight-forecast-hub Flusight/2023-2024/FluSight-forecast-hub-official
git clone https://github.com/cdcepi/FluSight-forecast-hub Flusight/2024-2025/FluSight-forecast-hub-official
git clone https://github.com/midas-network/flu-scenario-modeling-hub.git Flusight/Flu-SMH
git clone https://github.com/ACCIDDA/NC_Forecasting_Collab.git custom_datasets/NC_Forecasting_Collab
git clone https://github.com/adrian-lison/interval-scoring.git interval_scoring
```

## Update Data

```bash
./update-data.sh
```
