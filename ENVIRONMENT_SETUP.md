# InfluPaint Environment Setup Guide

This guide explains how to set up the development environment for InfluPaint using `uv`, a fast Python package installer and resolver.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended: NVIDIA V100, A100, or L40 with 16GB+ VRAM)
- Git
- [uv](https://github.com/astral-sh/uv) package installer

## Installing uv

If you don't have `uv` installed, install it with:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using pip
pip install uv
```

## Quick Setup with uv

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/influpaint.git
cd influpaint
```

### 2. Create Virtual Environment with uv

```bash
# Create a new virtual environment using uv
uv venv .venv

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install all dependencies from requirements.txt
uv pip install -r requirements.txt

# Install PyTorch with CUDA support (if you have an NVIDIA GPU)
# For CUDA 11.8:
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU-only (not recommended for this project):
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install External Dependencies

InfluPaint depends on external repositories that need to be cloned:

```bash
# Create directory for reference implementations
mkdir -p referenceimplementations

# Clone guided-diffusion (OpenAI's guided diffusion implementation)
git clone https://github.com/openai/guided-diffusion.git referenceimplementations/guided-diffusion

# Clone Delphi Epidata API
git clone https://github.com/cmu-delphi/delphi-epidata.git Flusight/flu-datasets/delphi-epidata

# Clone FluSight data repositories
git clone https://github.com/cdcepi/Flusight-forecast-data.git Flusight/2022-2023/FluSight-forecast-hub-official
git clone https://github.com/cdcepi/FluSight-forecast-hub Flusight/2023-2024/FluSight-forecast-hub-official
git clone https://github.com/cdcepi/FluSight-forecast-hub Flusight/2024-2025/FluSight-forecast-hub-official

# Clone Flu Scenario Modeling Hub
git clone https://github.com/midas-network/flu-scenario-modeling-hub.git Flusight/Flu-SMH

# Clone NC Forecasting Collaboration (if needed)
mkdir -p custom_datasets
git clone https://github.com/ACCIDDA/NC_Forecasting_Collab.git custom_datasets/NC_Forecasting_Collab
```

### 5. Add guided-diffusion to Python Path

Add the guided-diffusion directory to your Python path:

```bash
# On Linux/macOS, add to your .bashrc or .zshrc:
export PYTHONPATH="${PYTHONPATH}:$(pwd)/referenceimplementations/guided-diffusion"

# Or create a .pth file in your virtual environment
echo "$(pwd)/referenceimplementations/guided-diffusion" > .venv/lib/python*/site-packages/guided_diffusion.pth
```

### 6. Install Jupyter Kernel (Optional)

If you plan to use Jupyter notebooks:

```bash
# Install IPython kernel for this environment
python -m ipykernel install --user --name influpaint --display-name "Python (InfluPaint)"
```

### 7. Verify Installation

```bash
# Test Python imports
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import influpaint; print('InfluPaint imported successfully')"
```

## Package List

### Core Dependencies

| Package | Purpose |
|---------|---------|
| **numpy** | Numerical computing |
| **pandas** | Data manipulation and analysis |
| **scipy** | Scientific computing |
| **xarray** | Multi-dimensional labeled arrays |
| **netcdf4, h5netcdf** | NetCDF file I/O |

### Visualization

| Package | Purpose |
|---------|---------|
| **matplotlib** | Plotting library |
| **seaborn** | Statistical data visualization |
| **plotly** | Interactive plots |
| **forestplot** | Forest plot visualization |

### Deep Learning

| Package | Purpose |
|---------|---------|
| **torch** | PyTorch deep learning framework |
| **torchvision** | Computer vision utilities |
| **einops** | Tensor operations |

### CLI & Utilities

| Package | Purpose |
|---------|---------|
| **click** | Command-line interface creation |
| **tqdm** | Progress bars |
| **tenacity** | Retry utilities |
| **aiohttp** | Async HTTP client |

### Development Tools

| Package | Purpose |
|---------|---------|
| **jupyterlab** | Interactive development environment |
| **ipykernel** | Jupyter kernel |
| **ipywidgets** | Interactive widgets |
| **pytest** | Testing framework |

### Domain-Specific

| Package | Purpose |
|---------|---------|
| **epiweeks** | Epidemiological week calculations |
| **mlflow** | ML experiment tracking |
| **tabulate** | Table formatting |

### External Repositories

| Repository | Purpose |
|------------|---------|
| **guided-diffusion** | OpenAI's diffusion model implementation |
| **delphi-epidata** | CDC Delphi Epidata API |
| **FluSight-forecast-hub** | FluSight challenge data |
| **flu-scenario-modeling-hub** | Flu scenario modeling data |

## Updating Dependencies

To update all packages to their latest compatible versions:

```bash
# Update all packages
uv pip install --upgrade -r requirements.txt
```

To update data repositories:

```bash
# Use the provided update script
./update-data.sh
```

## Alternative: Using pyproject.toml (Recommended for Future)

For better dependency management, consider creating a `pyproject.toml`:

```bash
# Initialize a new project with uv
uv init

# This will create a pyproject.toml that you can customize
```

## HPC Environment (UNC Longleaf)

If you're on UNC Longleaf or similar HPC environments:

```bash
# Load required modules
module purge
module load python/3.9.6

# Install uv in user space
pip install --user uv

# Follow the setup steps above
```

## Troubleshooting

### Issue: CUDA not available

**Solution:** Ensure you have NVIDIA drivers installed and reinstall PyTorch with CUDA support:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
```

### Issue: Import errors for guided_diffusion

**Solution:** Make sure the guided-diffusion repository is cloned and added to your PYTHONPATH:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/referenceimplementations/guided-diffusion"
```

### Issue: MLflow tracking errors

**Solution:** Initialize MLflow tracking directory:

```bash
mkdir -p mlruns
```

## Running the Project

### Training

```bash
# Run training with default configuration
python influpaint/batch/training.py

# Or use the provided run file
# Edit train.run with your configuration, then:
sbatch train.run  # On SLURM systems
```

### Data Preparation

```bash
# Gather all flu datasets
python 1-gather_all_flu_datasets_ipynb.py

# Build training datasets
python 2-build_training_flu_datasets_ipynb.py
```

### Jupyter Notebooks

```bash
# Start Jupyter Lab
jupyter lab

# Or on HPC, use the provided script:
sh runjupyter.sh
```

## Additional Resources

- [uv Documentation](https://github.com/astral-sh/uv)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [InfluPaint Repository](https://github.com/YOUR_USERNAME/influpaint)
- [Original Paper/Documentation](./README.md)
