"""
Configuration and constants for paper figures generation.
"""

import os
import datetime as dt
import glob
import matplotlib.pyplot as plt


# State abbreviation to full name mapping
STATE_NAMES = {
    'CA': 'California', 'FL': 'Florida', 'MT': 'Montana', 'NC': 'North Carolina',
    'NY': 'New York', 'TX': 'Texas', 'IL': 'Illinois'
}

# Image dimensions
IMAGE_SIZE = 64
CHANNELS = 1

# Plotting options
PLOT_MEDIAN = True

# Trajectory filtering options
MAX_LOW_LOCATIONS = 10  # Max number of locations allowed to have peaks below threshold

# Model configuration
BEST_MODEL_ID = "i868"
BEST_CONFIG = "celebahq_noTTJ5"


def find_uncond_samples_path(model_id: str, base_dir: str = "from_longleaf/regen/samples_regen/") -> str:
    """Find the unconditional samples file for a given model ID."""
    pattern = os.path.join(base_dir, f"inverse_transformed_samples_{model_id}*.npy")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No inverse transformed samples found for model {model_id}")
    if len(matches) > 1:
        raise ValueError(f"Multiple samples found for model {model_id}: {matches}")
    return matches[0]


# Paths
UNCOND_SAMPLES_PATH = find_uncond_samples_path(BEST_MODEL_ID)
INPAINTING_BASE = (
    "from_longleaf/influpaint_res/07b44fa_paper-2025-07-22_inpainting_2025-07-27"
)

# Output directory
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Model number for file naming
_MODEL_NUM = BEST_MODEL_ID.lstrip('i') if isinstance(BEST_MODEL_ID, str) else str(BEST_MODEL_ID)

# Fixed x-limits per season for publication-friendly alignment
SEASON_XLIMS = {
    '2023-2024': (dt.datetime(2023, 10, 7), dt.datetime(2024, 6, 1)),
    '2024-2025': (dt.datetime(2024, 11, 16), dt.datetime(2025, 5, 31)),
}

# Toggle: also show pre-forecast ("past") segments of NPY forecasts
SHOW_NPY_PAST = True


# Global matplotlib settings for publication
def setup_matplotlib():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['legend.fontsize'] = 10


# Initialize matplotlib settings on import
setup_matplotlib()
