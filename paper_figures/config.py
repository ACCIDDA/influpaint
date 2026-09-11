"""
Configuration and constants for paper figures generation.
"""

import os
import datetime as dt
import glob
from pathlib import Path
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

# Same trajectory indices as before revision so the figure insets do not change.
FIGURE1_INSET_SAMPLE_INDICES = (0, 255, 510)

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
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = None
UNCOND_SAMPLES_PATH = REPOSITORY_ROOT / (
    "from_longleaf/regen/samples_regen/"
    "inverse_transformed_samples_i868::m_U500cRx1224::ds_30S70M::tr_Sqrt::ri_No.npy"
)
INPAINTING_BASE = REPOSITORY_ROOT / (
    "from_longleaf/influpaint_res/07b44fa_paper-2025-07-22_inpainting_2025-07-27"
)
HISTORICAL_OBSERVATIONS = REPOSITORY_ROOT / "influpaint/data/nhsn_flusight_past.csv"
LOCATION_FILE = REPOSITORY_ROOT / "influpaint/influpaint_locations.csv"
MASK_RESULTS_DIR = REPOSITORY_ROOT / "from_longleaf/mask_experiments_868_celebahq_noTTJ5"
GROUND_TRUTH_FILES = {
    season: REPOSITORY_ROOT / f"Flusight/{season}/FluSight-forecast-hub-official/target-data/target-hospital-admissions.csv"
    for season in ("2023-2024", "2024-2025")
}
MODEL_OUTPUT_DIRS = {
    season: REPOSITORY_ROOT / f"Flusight/{season}/FluSight-forecast-hub-official/model-output"
    for season in GROUND_TRUTH_FILES
}

# Output directory
FIG_DIR = "figures"


def use_archive(data_root):
    """Select the archived inputs for the existing paper plotting functions."""
    global ARCHIVE_ROOT, UNCOND_SAMPLES_PATH, INPAINTING_BASE
    global HISTORICAL_OBSERVATIONS, LOCATION_FILE, MASK_RESULTS_DIR
    global GROUND_TRUTH_FILES, MODEL_OUTPUT_DIRS
    ARCHIVE_ROOT = Path(data_root).resolve()
    UNCOND_SAMPLES_PATH = Path(find_uncond_samples_path(
        BEST_MODEL_ID, str(ARCHIVE_ROOT / "forecasts/unconditional")))
    INPAINTING_BASE = ARCHIVE_ROOT / "forecasts/retrospective"
    HISTORICAL_OBSERVATIONS = ARCHIVE_ROOT / "observations/nhsn_flusight_past.csv"
    LOCATION_FILE = ARCHIVE_ROOT / "observations/influpaint_locations.csv"
    MASK_RESULTS_DIR = ARCHIVE_ROOT / "forecasts/masks"
    GROUND_TRUTH_FILES = {
        season: ARCHIVE_ROOT / f"observations/{season}/target-hospital-admissions.csv"
        for season in ("2023-2024", "2024-2025")
    }
    MODEL_OUTPUT_DIRS = {
        season: ARCHIVE_ROOT / "forecasts/operational" / season
        for season in GROUND_TRUTH_FILES
    }

# Model number for file naming
_MODEL_NUM = BEST_MODEL_ID.lstrip('i') if isinstance(BEST_MODEL_ID, str) else str(BEST_MODEL_ID)

# Fixed x-limits per season for publication-friendly alignment
SEASON_XLIMS = {
    '2023-2024': (dt.datetime(2023, 10, 7), dt.datetime(2024, 6, 1)),
    '2024-2025': (dt.datetime(2024, 11, 16), dt.datetime(2025, 5, 31)),
}

# Toggle: also show pre-forecast ("past") segments of NPY forecasts
SHOW_NPY_PAST = False


# Global matplotlib settings for publication
def setup_matplotlib():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['legend.fontsize'] = 10


# Initialize matplotlib settings on import
setup_matplotlib()
