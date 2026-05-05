"""Root cause candidate analysis for Module B process anomaly detection.

Produces ranked root cause CANDIDATES based on SPC violations, anomaly
detector outputs, and process-step feature semantics.

CRITICAL DESIGN CONSTRAINT
───────────────────────────
This module NEVER asserts a confirmed root cause. All outputs are described
as candidates or suspects. Real fab root cause analysis requires corroboration
with recipe files, equipment tool logs, consumable change records, metrology
raw data, and engineer domain review. No automated system should stop a tool
or reject a lot based solely on this output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

# ── Synthetic process-step semantics ──────────────────────────────────────────

# Features measured at each known synthetic process step (from STEP_NOMINALS).
# Used to map feature anomalies back to the most likely originating step.
STEP_FEATURE_MAP: dict[str, list[str]] = {
    "Lithography": ["temperature", "pressure", "exposure_dose", "overlay_error", "defect_density"],
    "Etching":     ["temperature", "pressure", "gas_flow", "rf_power", "film_thickness", "defect_density"],
    "Deposition":  ["temperature", "pressure", "gas_flow", "rf_power", "film_thickness", "defect_density"],
    "CMP":         ["temperature", "pressure", "film_thickness", "defect_density"],
    "Metrology":   ["temperature", "pressure", "film_thickness", "overlay_error", "defect_density"],
}

# Features that uniquely distinguish one step from others (step-discriminating).
# An anomaly in a discriminating feature is stronger evidence against that step.
STEP_DISCRIMINATING_FEATURES: dict[str, list[str]] = {
    "Lithography": ["exposure_dose", "overlay_error"],
    "Etching":     ["gas_flow", "rf_power"],
    "Deposition":  ["gas_flow", "rf_power"],
    "CMP":         [],
    "Metrology":   ["overlay_error"],
}

# Known synthetic step names (used to guard SECOM anonymous mode)
KNOWN_SYNTHETIC_STEPS: frozenset[str] = frozenset(STEP_FEATURE_MAP.keys())

# ── Recommended engineer checks ────────────────────────────────────────────────

_STEP_CHECKS: dict[str, list[str]] = {
    "Lithography": [
        "Inspect scanner exposure dose delivery records for the flagged lots",
        "Review overlay metrology trend charts for registration drift",
        "Check track coater and developer temperature control logs",
        "Verify reticle inspection records and alignment mark condition",
    ],
    "Etching": [
        "Review RF power supply calibration logs for the etch chamber",
        "Inspect gas flow MFC calibration records and control chart trend",
        "Check chamber pressure readings against recipe specification",
        "Review etch rate uniformity from film thickness measurements",
        "Inspect endpoint detection signal for the affected lots",
    ],
    "Deposition": [
        "Check CVD/PVD chamber temperature uniformity records",
        "Review precursor gas flow MFC calibration",
        "Inspect deposited film thickness uniformity across wafer",
        "Review RF power delivery logs for plasma-enhanced deposition steps",
        "Check chamber conditioning log for maintenance events around the flagged period",
    ],
    "CMP": [
        "Review slurry flow rate, composition, and pH records",
        "Inspect polishing pad conditioning schedule and pad wear profile",
        "Check down-force and back-pressure control logs",
        "Review post-CMP film thickness uniformity for the affected lots",
    ],
    "Metrology": [
        "Verify measurement tool calibration against reference standards",
        "Assess whether anomaly reflects measurement noise vs. actual process signal",
        "Review defect density trend for contamination events or particle excursions",
        "Confirm measurement recipe settings were not changed around the flagged period",
    ],
}

_FEATURE_CHECKS: dict[str, str] = {
    "temperature":   "Review temperature controller setpoints, PID tuning, and sensor calibration for the affected step",
    "pressure":      "Check chamber pressure control system logs and vacuum gauge calibration",
    "gas_flow":      "Inspect MFC calibration records, gas line conditions, and mass flow controller logs",
    "rf_power":      "Review RF generator power delivery logs and matching network tuning records",
    "exposure_dose": "Check scanner dose monitor readings, integrator calibration, and light source energy records",
    "film_thickness":"Review deposition/etch rate trend and endpoint detection system for the affected lots",
    "overlay_error": "Inspect scanner alignment system logs, reticle condition, and wafer chuck calibration",
    "defect_density":"Review contamination events log, particle monitor data, and tool maintenance history",
}

LIMITATION_NOTE: str = (
    "This output lists root cause CANDIDATES derived from statistical patterns in process "
    "data only. It is NOT a confirmed or actionable root cause diagnosis. Real fab root "
    "cause analysis requires corroboration with recipe files, equipment tool logs, consumable "
    "change records, metrology raw data, and engineer domain review. No automated system "
    "should stop a tool or reject a lot based solely on this output."
)


# ── Output data structure ──────────────────────────────────────────────────────

@dataclass
class RCACandidate:
    """A single root cause candidate for one suspected process step.

    Attributes:
        suspected_process_step: Name of the step (synthetic) or anonymous ID (SECOM).
        suspicious_features: Features that showed statistical anomalies at this step.
        evidence: Human-readable evidence strings (SPC violations, anomaly fractions).
        recommended_checks: Specific engineer actions to triage this candidate.
        confidence_level: ``"low"`` / ``"medium"`` / ``"high"`` — reflects evidence breadth,
            not probability of being the true root cause.
        limitation_note: Always present. Reminds users this is a candidate, not a verdict.
        _score: Internal ranking score; not part of the public output.
    """

    suspected_process_step: str
    suspicious_features: list[str]
    evidence: list[str]
    recommended_checks: list[str]
    confidence_level: Literal["low", "medium", "high"]
    limitation_note: str
    _score: float = field(default=0.0, repr=False, compare=False)


# ── Core analysis ──────────────────────────────────────────────────────────────

def analyze(
    df: pd.DataFrame,
    spc_violations: pd.DataFrame | None = None,
    anomaly_scores: pd.DataFrame | None = None,
    feature_groups: dict[str, list[str]] | None = None,
    data_source: Literal["synthetic", "secom"] = "synthetic",
    top_n: int = 3,
) -> list[RCACandidate]:
    """Produce ranked root cause candidates from process data and analysis outputs.

    Args:
        df: Full process DataFrame. Must contain ``process_step`` and ``lot_id``
            columns for synthetic data.
        spc_violations: Violations DataFrame from ``spc.run_spc()``. Expected
            columns: ``process_step``, ``feature``, ``rule``, ``severity``.
            Pass ``None`` to skip SPC evidence.
        anomaly_scores: Row-level anomaly scores, typically from
            ``outputs/reports/process/anomaly_scores.csv``. Expected columns:
            ``process_step``, ``ensemble_any`` (or ``if_pred`` / ``ae_pred``).
            Pass ``None`` to skip anomaly detector evidence.
        feature_groups: Optional mapping of group label → feature list, used
            to annotate evidence with group context (e.g. ``{"sensor": [...],
            "metrology": [...]}``. Has no effect on scoring.
        data_source: ``"synthetic"`` allows named process steps in output;
            ``"secom"`` suppresses all known fab step names to avoid misleading
            claims about anonymous features.
        top_n: Maximum number of candidates to return, sorted by score.

    Returns:
        List of :class:`RCACandidate` sorted by descending score (highest
        suspicion first). May be shorter than ``top_n`` if fewer steps have
        evidence.
    """
    if data_source == "secom":
        return _analyze_secom(df, spc_violations, anomaly_scores, feature_groups, top_n)
    return _analyze_synthetic(df, spc_violations, anomaly_scores, feature_groups, top_n)


# ── Synthetic data path ────────────────────────────────────────────────────────

def _analyze_synthetic(
    df: pd.DataFrame,
    spc_violations: pd.DataFrame | None,
    anomaly_scores: pd.DataFrame | None,
    feature_groups: dict[str, list[str]] | None,
    top_n: int,
) -> list[RCACandidate]:
    steps = (
        df["process_step"].unique().tolist()
        if "process_step" in df.columns
        else list(STEP_FEATURE_MAP.keys())
    )

    candidates: list[RCACandidate] = []

    for step in steps:
        step_rows = df[df["process_step"] == step] if "process_step" in df.columns else df
        n_step_rows = max(len(step_rows), 1)

        # ── SPC evidence ──────────────────────────────────────────────────────
        spc_features: list[str] = []
        spc_evidence: list[str] = []
        n_spc_violations_total = 0

        if spc_violations is not None and len(spc_violations) > 0:
            step_viols = spc_violations[spc_violations["process_step"] == step]
            if len(step_viols) > 0:
                n_spc_violations_total = len(step_viols)
                by_feature = step_viols.groupby("feature")
                for feat, grp in by_feature:
                    spc_features.append(str(feat))
                    rules = sorted(grp["rule"].unique())
                    severity = grp["severity"].iloc[0] if "severity" in grp.columns else ""
                    spc_evidence.append(
                        f"SPC: {feat} — {len(grp)} violation(s) at {step} "
                        f"(rules: {', '.join(rules)}"
                        + (f", severity: {severity}" if severity else "")
                        + ")"
                    )

        # ── Anomaly detector evidence ─────────────────────────────────────────
        anomaly_fraction = 0.0
        anomaly_evidence: list[str] = []

        if anomaly_scores is not None and len(anomaly_scores) > 0:
            pred_col = _best_pred_col(anomaly_scores)
            if pred_col and "process_step" in anomaly_scores.columns:
                step_scores = anomaly_scores[anomaly_scores["process_step"] == step]
                if len(step_scores) > 0:
                    n_flagged = int(step_scores[pred_col].sum())
                    anomaly_fraction = n_flagged / len(step_scores)
                    if n_flagged > 0:
                        anomaly_evidence.append(
                            f"Anomaly detector ({pred_col}): {n_flagged}/{len(step_scores)} rows "
                            f"at {step} flagged ({anomaly_fraction:.1%})"
                        )

        # Skip steps with no evidence at all
        if not spc_features and anomaly_fraction == 0.0:
            continue

        # ── Scoring ───────────────────────────────────────────────────────────
        n_spc_features = len(set(spc_features))

        # Base: SPC feature breadth
        spc_score = n_spc_features * 3.0 + n_spc_violations_total * 0.3

        # Anomaly detector corroboration
        anomaly_score = anomaly_fraction * 10.0

        # Synergy bonus when both evidence sources agree
        synergy = 1.5 if (spc_score > 0 and anomaly_score > 0) else 1.0

        # Discriminating feature bonus: step-specific features are stronger evidence
        step_disc = STEP_DISCRIMINATING_FEATURES.get(step, [])
        n_disc_hit = sum(1 for f in spc_features if f in step_disc)
        disc_mult = 1.0 + 0.3 * n_disc_hit

        # Multi-feature consistency bonus
        multi_mult = 1.5 if n_spc_features >= 2 else 1.0

        score = (spc_score + anomaly_score) * synergy * disc_mult * multi_mult

        # ── Confidence ────────────────────────────────────────────────────────
        n_suspicious = max(n_spc_features, 1 if anomaly_fraction > 0 else 0)
        if n_suspicious >= 3 or (n_suspicious >= 2 and anomaly_fraction >= 0.05):
            confidence: Literal["low", "medium", "high"] = "high"
        elif n_suspicious >= 2 or (n_spc_features >= 1 and anomaly_fraction >= 0.05):
            confidence = "medium"
        else:
            confidence = "low"

        # ── Recommended checks ────────────────────────────────────────────────
        checks = list(_STEP_CHECKS.get(step, []))
        for feat in set(spc_features):
            feat_check = _FEATURE_CHECKS.get(feat)
            if feat_check and feat_check not in checks:
                checks.append(feat_check)
        if not checks:
            checks = ["Review process logs and metrology data for the affected step and lots"]

        all_evidence = spc_evidence + anomaly_evidence
        if not all_evidence:
            all_evidence = ["No direct SPC or anomaly evidence recorded for this step"]

        candidates.append(
            RCACandidate(
                suspected_process_step=step,
                suspicious_features=sorted(set(spc_features)),
                evidence=all_evidence,
                recommended_checks=checks[:6],  # cap at 6 items
                confidence_level=confidence,
                limitation_note=LIMITATION_NOTE,
                _score=score,
            )
        )

    candidates.sort(key=lambda c: c._score, reverse=True)
    return candidates[:top_n]


# ── SECOM anonymous data path ──────────────────────────────────────────────────

def _analyze_secom(
    df: pd.DataFrame,
    spc_violations: pd.DataFrame | None,
    anomaly_scores: pd.DataFrame | None,
    feature_groups: dict[str, list[str]] | None,
    top_n: int,
) -> list[RCACandidate]:
    """SECOM mode: anonymous features — no real fab step names in output.

    Groups anomalous features by occurrence frequency and outputs them as
    anonymous clusters. Known synthetic step names are explicitly excluded
    from candidate labels to prevent misleading claims about anonymous data.
    """
    # Build anonymous groups from SPC violations
    feature_counts: dict[str, int] = {}
    feature_evidence: dict[str, list[str]] = {}

    if spc_violations is not None and len(spc_violations) > 0:
        for feat, grp in spc_violations.groupby("feature"):
            feat = str(feat)
            if feat in KNOWN_SYNTHETIC_STEPS:
                continue
            feature_counts[feat] = feature_counts.get(feat, 0) + len(grp)
            rules = sorted(grp["rule"].unique())
            feature_evidence.setdefault(feat, []).append(
                f"SPC: {feat} — {len(grp)} violation(s) (rules: {', '.join(rules)})"
            )

    if anomaly_scores is not None and len(anomaly_scores) > 0:
        pred_col = _best_pred_col(anomaly_scores)
        if pred_col:
            n_flagged = int(anomaly_scores[pred_col].sum())
            total = len(anomaly_scores)
            if n_flagged > 0:
                feature_evidence.setdefault("__anomaly_detector__", []).append(
                    f"Anomaly detector ({pred_col}): {n_flagged}/{total} samples flagged "
                    f"({n_flagged / total:.1%})"
                )

    if not feature_counts and not feature_evidence:
        return []

    # Sort features by violation count
    sorted_feats = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
    top_feats = [f for f, _ in sorted_feats[:8]]

    # Group into a single anonymous candidate (no step-level semantics for SECOM)
    all_evidence: list[str] = []
    for feat in top_feats:
        all_evidence.extend(feature_evidence.get(feat, []))
    all_evidence.extend(feature_evidence.get("__anomaly_detector__", []))

    n_anom_feats = len(top_feats)
    if n_anom_feats >= 3:
        confidence: Literal["low", "medium", "high"] = "medium"
    else:
        confidence = "low"

    secom_note = (
        LIMITATION_NOTE + " Additionally, feature names in this dataset are anonymous — "
        "mapping anomalies to specific process steps is not possible without the "
        "proprietary feature index mapping held by the original data provider."
    )

    candidates = [
        RCACandidate(
            suspected_process_step=f"anonymous_cluster_{i + 1}",
            suspicious_features=top_feats[i * 3 : (i + 1) * 3],
            evidence=all_evidence[i * 2 : (i + 1) * 2] if all_evidence else ["No direct SPC evidence"],
            recommended_checks=[
                "Cross-reference flagged anonymous features with the original SECOM feature index documentation",
                "Review lot yield trend for the affected time window",
                "Compare flagged feature distributions between pass and fail lots",
                "Engage process engineers who have access to the feature-to-parameter mapping",
            ],
            confidence_level=confidence,
            limitation_note=secom_note,
            _score=float(n_anom_feats - i),
        )
        for i in range(min(top_n, max(1, (len(top_feats) + 2) // 3)))
    ]

    return candidates


# ── Helpers ────────────────────────────────────────────────────────────────────

def _best_pred_col(anomaly_scores: pd.DataFrame) -> str | None:
    """Return the best available binary prediction column from anomaly_scores."""
    for col in ("ensemble_any", "if_pred", "ae_pred"):
        if col in anomaly_scores.columns:
            return col
    return None
