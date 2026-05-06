"""Statistical Process Control (SPC) with Western Electric Rules.

SPC signals indicate that a process parameter has deviated from its expected
statistical behavior. A violation is a prompt for engineer investigation —
NOT a confirmed defect or root cause. False-alarm rates of 5-15% are normal
even in a stable process.

Reference: Western Electric Statistical Quality Control Handbook (1956);
           ISO 7870-2 Control Charts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

# ── Rule catalog ───────────────────────────────────────────────────────────────

WE_RULES: dict[str, str] = {
    "Rule 1": "One point beyond 3σ",
    "Rule 2": "Two of three consecutive points beyond 2σ on same side",
    "Rule 3": "Four of five consecutive points beyond 1σ on same side",
    "Rule 4": "Eight consecutive points on same side of center line",
}

RULE_SEVERITY: dict[str, str] = {
    "Rule 1": "HIGH",
    "Rule 2": "MEDIUM",
    "Rule 3": "LOW",
    "Rule 4": "LOW",
}


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ControlLimits:
    """Control limits for a single (feature, process-step) pair.

    Limits are computed from the data itself (Phase-1 approach).
    In production, limits would be derived from a separate stable-process
    baseline period to avoid contamination by the anomalies under study.
    """

    mean: float
    std: float

    @property
    def ucl(self) -> float:
        return self.mean + 3 * self.std

    @property
    def lcl(self) -> float:
        return self.mean - 3 * self.std

    @property
    def ucl_2(self) -> float:
        return self.mean + 2 * self.std

    @property
    def lcl_2(self) -> float:
        return self.mean - 2 * self.std

    @property
    def ucl_1(self) -> float:
        return self.mean + self.std

    @property
    def lcl_1(self) -> float:
        return self.mean - self.std


@dataclass
class SPCViolation:
    """One triggered WE rule at one point in time."""

    timestamp: pd.Timestamp
    feature: str
    rule: str
    rule_description: str
    value: float
    severity: str
    index: int  # position within the sorted group series


# ── Core calculations ──────────────────────────────────────────────────────────

def compute_control_limits(values: np.ndarray) -> ControlLimits:
    """Compute mean ± 3σ control limits from a 1-D measurement array.

    NaN entries are ignored. Raises ValueError if fewer than 2 valid points
    remain (cannot estimate std from a single observation).
    """
    clean = values[~np.isnan(values)]
    if len(clean) < 2:
        raise ValueError(
            f"Need at least 2 non-NaN values to compute control limits, got {len(clean)}"
        )
    return ControlLimits(mean=float(np.mean(clean)), std=float(np.std(clean, ddof=1)))


def check_western_electric_rules(
    values: np.ndarray,
    timestamps: np.ndarray,
    limits: ControlLimits,
    feature: str,
    deduplicate_rule4: bool = False,
) -> list[SPCViolation]:
    """Apply all four WE rules and return every triggered violation.

    Each returned SPCViolation carries the triggering point's index within
    `values`, so it can be marked directly on a control chart.

    Args:
        deduplicate_rule4: When True, uses _rule4_events instead of _rule4.
            _rule4 fires at every index where the 8-point sliding window
            qualifies — producing n-7 violations for a run of n consecutive
            same-side points.  _rule4_events fires only once per distinct run,
            at the 8th point.  Use deduplicate_rule4=True for event counting;
            keep False (default) for chart annotation so every out-of-control
            point is visually marked.
    """
    violations: list[SPCViolation] = []
    violations.extend(_rule1(values, timestamps, limits, feature))
    violations.extend(_rule2(values, timestamps, limits, feature))
    violations.extend(_rule3(values, timestamps, limits, feature))
    rule4_fn = _rule4_events if deduplicate_rule4 else _rule4
    violations.extend(rule4_fn(values, timestamps, limits, feature))
    return violations


def run_spc(
    df: pd.DataFrame,
    feature_cols: list[str],
    timestamp_col: str = "timestamp",
    group_col: str | None = "process_step",
    deduplicate_rule4: bool = False,
) -> tuple[pd.DataFrame, dict[tuple[str, str], ControlLimits]]:
    """Run SPC across all features, optionally grouped by process step.

    Args:
        df: Process data DataFrame (one row per measurement).
        feature_cols: Numeric columns to monitor with SPC.
        timestamp_col: Column used for ordering within each group and
                       stored in the output violation table.
        group_col: Column to group by before computing limits (e.g.
                   ``"process_step"``). Pass ``None`` to treat all rows as one
                   group. Each group gets its own independent control limits.

    Returns:
        Tuple of:
          - **violations_df**: One row per SPC signal, sorted by timestamp.
            Columns: timestamp, feature, process_step, rule, rule_description,
            value, severity, series_index.
          - **limits_map**: ``{(feature, group_str): ControlLimits}`` for every
            (feature, group) pair that had enough non-NaN data.
    """
    all_violations: list[dict] = []
    limits_map: dict[tuple[str, str], ControlLimits] = {}

    if group_col is None:
        groups: list[tuple[str, pd.DataFrame]] = [("all", df)]
    else:
        groups = [(str(name), grp) for name, grp in df.groupby(group_col)]

    for group_key, group_df in groups:
        group_df = group_df.sort_values(timestamp_col).reset_index(drop=True)
        timestamps = group_df[timestamp_col].values

        for feature in feature_cols:
            if feature not in group_df.columns:
                continue
            values = group_df[feature].values.astype(float)
            if (~np.isnan(values)).sum() < 2:
                continue

            limits = compute_control_limits(values)
            limits_map[(feature, group_key)] = limits

            for v in check_western_electric_rules(
                values, timestamps, limits, feature,
                deduplicate_rule4=deduplicate_rule4,
            ):
                all_violations.append(
                    {
                        "timestamp":       v.timestamp,
                        "feature":         v.feature,
                        "feature_name":    v.feature,           # alias for spec compliance
                        "process_step":    group_key,
                        "rule":            v.rule,
                        "rule_name":       v.rule,              # alias for spec compliance
                        "rule_description": v.rule_description,
                        "value":           round(float(v.value), 6),
                        "severity":        v.severity,
                        "mean":            round(limits.mean, 6),
                        "ucl":             round(limits.ucl, 6),
                        "lcl":             round(limits.lcl, 6),
                        "series_index":    v.index,
                    }
                )

    if all_violations:
        violations_df = (
            pd.DataFrame(all_violations)
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
    else:
        violations_df = pd.DataFrame(
            columns=[
                "timestamp", "feature", "feature_name", "process_step",
                "rule", "rule_name", "rule_description",
                "value", "severity", "mean", "ucl", "lcl", "series_index",
            ]
        )

    logger.info(
        f"[SPC] {len(violations_df)} violations | "
        f"{len(feature_cols)} features | "
        f"{len(limits_map)} (feature, step) groups monitored"
    )
    return violations_df, limits_map


# ── Private rule implementations ───────────────────────────────────────────────

def _viol(
    i: int,
    values: np.ndarray,
    timestamps: np.ndarray,
    feature: str,
    rule: str,
) -> SPCViolation:
    return SPCViolation(
        timestamp=pd.Timestamp(timestamps[i]),
        feature=feature,
        rule=rule,
        rule_description=WE_RULES[rule],
        value=float(values[i]),
        severity=RULE_SEVERITY[rule],
        index=i,
    )


def _rule1(
    values: np.ndarray,
    timestamps: np.ndarray,
    limits: ControlLimits,
    feature: str,
) -> list[SPCViolation]:
    """One point beyond 3σ."""
    result = []
    for i, v in enumerate(values):
        if not np.isnan(v) and (v > limits.ucl or v < limits.lcl):
            result.append(_viol(i, values, timestamps, feature, "Rule 1"))
    return result


def _rule2(
    values: np.ndarray,
    timestamps: np.ndarray,
    limits: ControlLimits,
    feature: str,
) -> list[SPCViolation]:
    """Two of three consecutive points beyond 2σ on same side."""
    result = []
    n = len(values)
    for i in range(2, n):
        w = values[i - 2 : i + 1]
        if np.any(np.isnan(w)):
            continue
        above = int(np.sum(w > limits.ucl_2))
        below = int(np.sum(w < limits.lcl_2))
        if above >= 2 or below >= 2:
            result.append(_viol(i, values, timestamps, feature, "Rule 2"))
    return result


def _rule3(
    values: np.ndarray,
    timestamps: np.ndarray,
    limits: ControlLimits,
    feature: str,
) -> list[SPCViolation]:
    """Four of five consecutive points beyond 1σ on same side."""
    result = []
    n = len(values)
    for i in range(4, n):
        w = values[i - 4 : i + 1]
        if np.any(np.isnan(w)):
            continue
        above = int(np.sum(w > limits.ucl_1))
        below = int(np.sum(w < limits.lcl_1))
        if above >= 4 or below >= 4:
            result.append(_viol(i, values, timestamps, feature, "Rule 3"))
    return result


def _rule4(
    values: np.ndarray,
    timestamps: np.ndarray,
    limits: ControlLimits,
    feature: str,
) -> list[SPCViolation]:
    """Eight consecutive points on same side of center line.

    Fires once per sliding window that qualifies.  A sustained run of n
    same-side points produces n-7 violations — use _rule4_events when
    counting distinct drift events rather than marking every out-of-control
    point on a chart.
    """
    result = []
    n = len(values)
    for i in range(7, n):
        w = values[i - 7 : i + 1]
        if np.any(np.isnan(w)):
            continue
        if np.all(w > limits.mean) or np.all(w < limits.mean):
            result.append(_viol(i, values, timestamps, feature, "Rule 4"))
    return result


def _rule4_events(
    values: np.ndarray,
    timestamps: np.ndarray,
    limits: ControlLimits,
    feature: str,
) -> list[SPCViolation]:
    """Rule 4 (deduplicated): one SPCViolation per distinct qualifying run.

    Fires at the 8th consecutive same-side point of each new run and does NOT
    re-fire for extensions of that run.  A run resets when a point falls
    exactly on the mean, a NaN appears, or the side flips.

    This avoids the inflated counts produced by _rule4 when a sustained drift
    keeps the process on one side of the center line for many consecutive points
    (e.g. a 250-point drift yields 243 raw Rule 4 violations but only 1 event).
    """
    result: list[SPCViolation] = []
    run_len = 0
    run_above: bool | None = None
    event_fired = False

    for i, v in enumerate(values):
        if np.isnan(v):
            run_len = 0
            run_above = None
            event_fired = False
            continue

        current_above = v > limits.mean
        current_below = v < limits.mean

        if not (current_above or current_below):
            # Exactly at mean — breaks any run
            run_len = 0
            run_above = None
            event_fired = False
            continue

        current_side = current_above  # True = above, False = below

        if run_above is None or current_side != run_above:
            run_len = 1
            run_above = current_side
            event_fired = False
        else:
            run_len += 1

        if run_len >= 8 and not event_fired:
            result.append(_viol(i, values, timestamps, feature, "Rule 4"))
            event_fired = True  # suppress further fires until run breaks

    return result
