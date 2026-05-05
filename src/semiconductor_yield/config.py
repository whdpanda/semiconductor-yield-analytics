"""Central path and runtime configuration.

All other modules import paths from here — never hardcode absolute paths elsewhere.
"""

from pathlib import Path

# ── Project root (three levels up from this file: src/semiconductor_yield/config.py) ──
ROOT_DIR = Path(__file__).parent.parent.parent.resolve()

# ── Data directories ───────────────────────────────────────────────────────────
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"

# ── Raw dataset paths ──────────────────────────────────────────────────────────
WM811K_PKL = RAW_DIR / "wm811k" / "LSWMD.pkl"
SECOM_DIR = RAW_DIR / "secom"

# ── Processed data paths ───────────────────────────────────────────────────────
WAFER_MAPS_DIR = PROCESSED_DIR / "wafer_maps"
SPC_PROCESSED_DIR = PROCESSED_DIR / "spc"

# ── Output directories ─────────────────────────────────────────────────────────
OUTPUTS_DIR = ROOT_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"
GRADCAM_DIR = REPORTS_DIR / "gradcam"
WAFER_REPORTS_DIR = REPORTS_DIR / "wafer"
PROCESS_REPORTS_DIR = REPORTS_DIR / "process"
LOGS_DIR = OUTPUTS_DIR / "logs"

# ── Config directory ───────────────────────────────────────────────────────────
CONFIGS_DIR = ROOT_DIR / "configs"
MODULE_A_CONFIG = CONFIGS_DIR / "module_a.yaml"
MODULE_B_CONFIG = CONFIGS_DIR / "module_b.yaml"

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Wafer map settings ─────────────────────────────────────────────────────────
WAFER_MAP_SIZE = (64, 64)
WAFER_DEFECT_CLASSES = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "none",
]
NUM_WAFER_CLASSES = len(WAFER_DEFECT_CLASSES)


def ensure_output_dirs() -> None:
    """Create all output directories if they don't exist."""
    for d in [MODELS_DIR, REPORTS_DIR, GRADCAM_DIR, WAFER_REPORTS_DIR, PROCESS_REPORTS_DIR, LOGS_DIR, WAFER_MAPS_DIR, SPC_PROCESSED_DIR]:
        d.mkdir(parents=True, exist_ok=True)
