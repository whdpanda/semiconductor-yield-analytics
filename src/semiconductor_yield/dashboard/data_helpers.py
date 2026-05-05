"""Pure data-transformation helpers for the analytics dashboard.

Intentionally free of Streamlit so these functions can be unit-tested
without a running Streamlit server.

Public API:
  load_spc_violations(path)        -> pd.DataFrame | None
  load_anomaly_scores(path)        -> pd.DataFrame | None
  load_anomaly_summary(path)       -> dict | None
  load_process_data(path)          -> pd.DataFrame | None
  get_violation_summary(df)        -> pd.DataFrame
  get_anomaly_rate_by_step(df)     -> pd.DataFrame
  get_available_charts(charts_dir) -> dict[str, Path]
  format_rca_candidates(candidates)-> list[dict]
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# ── File loaders ───────────────────────────────────────────────────────────────

def load_spc_violations(path: Path | str) -> pd.DataFrame | None:
    """Load spc_violations.csv.  Returns None if the file does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def load_anomaly_scores(path: Path | str) -> pd.DataFrame | None:
    """Load anomaly_scores.csv.  Returns None if the file does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def load_anomaly_summary(path: Path | str) -> dict | None:
    """Load anomaly_summary.json.  Returns None if the file does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_process_data(path: Path | str) -> pd.DataFrame | None:
    """Load process_data.csv.  Returns None if the file does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


# ── Data transformations ───────────────────────────────────────────────────────

def get_violation_summary(violations: pd.DataFrame | None) -> pd.DataFrame:
    """Summarise SPC violations grouped by process_step and rule.

    Args:
        violations: DataFrame with at least columns
            ``process_step``, ``rule``, ``feature``.
            None or empty DataFrame returns an empty summary.

    Returns:
        DataFrame with columns: process_step, rule, count, features.
        Sorted by process_step (asc) then count (desc).
    """
    empty = pd.DataFrame(columns=["process_step", "rule", "count", "features"])
    if violations is None or violations.empty:
        return empty
    required = {"process_step", "rule", "feature"}
    if not required.issubset(violations.columns):
        return empty

    grp = (
        violations
        .groupby(["process_step", "rule"], sort=True)
        .agg(
            count=("feature", "count"),
            features=("feature", lambda s: ", ".join(sorted(s.unique()))),
        )
        .reset_index()
        .sort_values(["process_step", "count"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return grp


def get_anomaly_rate_by_step(scores: pd.DataFrame | None) -> pd.DataFrame:
    """Return ensemble anomaly rate per process step.

    Prefers the ``ensemble_any`` column; falls back to ``if_pred``.

    Args:
        scores: DataFrame with at least columns
            ``process_step`` and one of ``ensemble_any`` / ``if_pred``.

    Returns:
        DataFrame with columns: process_step, total, flagged, rate_pct.
        Sorted by rate_pct descending.
    """
    empty = pd.DataFrame(columns=["process_step", "total", "flagged", "rate_pct"])
    if scores is None or scores.empty:
        return empty
    if "process_step" not in scores.columns:
        return empty

    flag_col = (
        "ensemble_any" if "ensemble_any" in scores.columns
        else "if_pred" if "if_pred" in scores.columns
        else None
    )
    if flag_col is None:
        return empty

    grp = (
        scores
        .groupby("process_step")[flag_col]
        .agg(total="count", flagged="sum")
        .reset_index()
    )
    grp["rate_pct"] = (grp["flagged"] / grp["total"] * 100).round(1)
    return grp.sort_values("rate_pct", ascending=False).reset_index(drop=True)


def get_available_charts(charts_dir: Path | str) -> dict[str, Path]:
    """Return a mapping of display label → PNG path for available SPC charts.

    Expected filename format: ``{STEP}__{FEATURE}.png``
    (double underscore separator, e.g. ``Etching__gas_flow.png``).
    Files that don't follow this convention use the stem as the label.

    Returns:
        Dict sorted by label. Empty dict if the directory does not exist.
    """
    charts_dir = Path(charts_dir)
    if not charts_dir.exists():
        return {}
    result: dict[str, Path] = {}
    for png in sorted(charts_dir.glob("*.png")):
        stem = png.stem
        if "__" in stem:
            step, feature = stem.split("__", 1)
            label = f"{step} | {feature}"
        else:
            label = stem
        result[label] = png
    return result


def format_rca_candidates(candidates: list) -> list[dict]:
    """Flatten RCACandidate objects into a list of dicts for tabular display.

    Args:
        candidates: List of ``RCACandidate`` dataclass instances.

    Returns:
        List of dicts with keys:
        rank, suspected_step, confidence, features, evidence_count.
    """
    result = []
    for rank, c in enumerate(candidates, 1):
        features = (
            ", ".join(c.suspicious_features)
            if c.suspicious_features
            else "—"
        )
        result.append({
            "rank":           rank,
            "suspected_step": c.suspected_process_step,
            "confidence":     c.confidence_level.upper(),
            "features":       features,
            "evidence_count": len(c.evidence),
        })
    return result
