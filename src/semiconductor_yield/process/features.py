"""Process feature definitions for Module B anomaly detection.

Feature groups are declared here so every script and test imports from one
place, preventing silent inconsistencies and making target-leakage risks
explicit and auditable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Feature group definitions ──────────────────────────────────────────────────

SENSOR_FEATURES: list[str] = [
    "temperature",
    "pressure",
    "gas_flow",
    "rf_power",
]

METROLOGY_FEATURES: list[str] = [
    "exposure_dose",
    "film_thickness",
    "overlay_error",
    "defect_density",
]

# yield_rate is generated as a function of anomaly_label in the synthetic
# generator (anomalous rows get lower yield). Using it as a detector input
# would constitute target leakage — the detector would be predicting yield
# from yield rather than detecting physics-based anomalies.
YIELD_FEATURES: list[str] = ["yield_rate"]

# Ground-truth columns (simulated data only). Must never appear as input features.
LABEL_COLS: list[str] = ["anomaly_label", "anomaly_type", "suspected_root_cause"]

IDENTIFIER_COLS: list[str] = ["lot_id", "wafer_id", "process_step", "timestamp"]

# ── Feature sets by temporal availability ──────────────────────────────────────

# In-situ process control parameters: reported by the process tool in real time
# during step execution. Available for anomaly detection *before* any post-process
# inspection is scheduled. Use this set when the model is intended to flag issues
# while the lot is still on the tool (earliest possible intervention).
#
# exposure_dose is included here because the lithography scanner reports delivered
# dose continuously — it is a recipe target, not a post-process measurement.
PROCESS_ONLY_FEATURES: list[str] = SENSOR_FEATURES + ["exposure_dose"]

# Adds the three offline inspection results measured *after* the process step
# completes: film thickness (ellipsometer/profilometer), overlay error (overlay
# metrology tool), and defect density (wafer inspection scanner). These are not
# available during process execution. Use this set for post-process quality
# monitoring models or retrospective anomaly analysis.
PROCESS_PLUS_METROLOGY_FEATURES: list[str] = PROCESS_ONLY_FEATURES + [
    "film_thickness",
    "overlay_error",
    "defect_density",
]

# ANOMALY_INPUT_FEATURES: backward-compatible alias for the full 8-feature set.
# Equivalent to PROCESS_PLUS_METROLOGY_FEATURES (= SENSOR_FEATURES + METROLOGY_FEATURES).
ANOMALY_INPUT_FEATURES: list[str] = PROCESS_PLUS_METROLOGY_FEATURES

# Map from CLI --feature-set argument to the corresponding list.
FEATURE_SET_MAP: dict[str, list[str]] = {
    "process_only": PROCESS_ONLY_FEATURES,
    "full": PROCESS_PLUS_METROLOGY_FEATURES,
}


def lot_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a process DataFrame into train / validation / test by lot_id.

    All rows belonging to a given lot stay in the same split, so no wafer
    from the same lot appears in both training and held-out evaluation data.

    Args:
        df: DataFrame with a ``lot_id`` column.
        train_frac: Fraction of lots for training (default 0.70).
        val_frac: Fraction of lots for validation (default 0.15).
        test_frac: Fraction of lots for held-out test (default 0.15).
        random_state: Seed for reproducible lot shuffling.

    Returns:
        ``(train_df, val_df, test_df)`` — disjoint row subsets of ``df``,
        each with a reset integer index.

    Raises:
        ValueError: If ``lot_id`` column is absent, fractions do not sum to 1,
            any fraction is non-positive, or there are too few lots to fill all
            three splits.
    """
    if "lot_id" not in df.columns:
        raise ValueError("DataFrame must contain a 'lot_id' column for lot-based splitting.")
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"train_frac + val_frac + test_frac must sum to 1.0, got {total:.6f}."
        )
    if any(f <= 0.0 for f in (train_frac, val_frac, test_frac)):
        raise ValueError("All split fractions must be positive.")

    lots = np.array(sorted(df["lot_id"].unique()))
    rng = np.random.default_rng(random_state)
    rng.shuffle(lots)

    n = len(lots)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    n_test  = n - n_train - n_val  # remainder avoids rounding loss

    if n_train < 1 or n_val < 1 or n_test < 1:
        raise ValueError(
            f"Not enough lots ({n}) to populate all three splits with fractions "
            f"{train_frac}/{val_frac}/{test_frac}. Need at least 3 lots."
        )

    train_lots = set(lots[:n_train])
    val_lots   = set(lots[n_train : n_train + n_val])
    test_lots  = set(lots[n_train + n_val :])

    return (
        df[df["lot_id"].isin(train_lots)].reset_index(drop=True),
        df[df["lot_id"].isin(val_lots)].reset_index(drop=True),
        df[df["lot_id"].isin(test_lots)].reset_index(drop=True),
    )


def get_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str] = ANOMALY_INPUT_FEATURES,
) -> tuple[np.ndarray, list[str]]:
    """Extract a (n, p) float64 feature matrix from a process DataFrame.

    Missing values arise because some parameters are not applicable for certain
    process steps (e.g. ``rf_power`` is NaN for Lithography rows). They are
    imputed with the column mean across all rows where the feature is non-NaN.
    This is a practical compromise for cross-step models; per-step detectors
    would eliminate NaN entirely and are more rigorous in production.

    Args:
        df: Process data DataFrame with at least the columns in ``feature_cols``.
        feature_cols: Columns to include. Missing columns are silently skipped.

    Returns:
        ``(X, valid_cols)`` where ``X`` is shape ``(len(df), len(valid_cols))``
        and ``valid_cols`` are the columns actually present in ``df``.

    Raises:
        ValueError: If no feature columns are present in ``df``.
    """
    present = [c for c in feature_cols if c in df.columns]
    if not present:
        raise ValueError(
            f"None of the requested feature columns are in the DataFrame. "
            f"Requested: {feature_cols}. Available: {list(df.columns)}"
        )

    X = df[present].values.astype(np.float64)

    # Impute column-wise: missing → column mean (fall back to 0 if all NaN)
    with np.errstate(all="ignore"):  # silence nanmean warning on all-NaN columns
        col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    for j, mean_val in enumerate(col_means):
        nan_mask = np.isnan(X[:, j])
        if nan_mask.any():
            X[nan_mask, j] = mean_val

    return X, present
