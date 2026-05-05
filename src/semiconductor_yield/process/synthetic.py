"""Synthetic semiconductor process data generator.

IMPORTANT: All generated data is SIMULATED. It does not represent real fab measurements,
real process recipes, or real equipment behavior. It exists solely to demonstrate SPC and
anomaly-detection pipelines in a portfolio context.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from semiconductor_yield.config import RANDOM_SEED

# ── Process step definitions ───────────────────────────────────────────────────

PROCESS_STEPS: list[str] = ["Lithography", "Etching", "Deposition", "CMP", "Metrology"]

VALID_ANOMALY_TYPES: frozenset[str] = frozenset(
    {"normal", "drift", "spike", "step_shift", "tool_offset"}
)

_NAN = float("nan")

# (mean, std) per (step, parameter).  NaN mean → parameter not applicable for this step.
STEP_NOMINALS: dict[str, dict[str, tuple[float, float]]] = {
    "Lithography": {
        "temperature":    (23.0,       0.30),   # °C  — track/coater environment
        "pressure":       (760_000.0,  500.0),  # mTorr — atmospheric for track
        "gas_flow":       (_NAN,       0.0),
        "rf_power":       (_NAN,       0.0),
        "exposure_dose":  (25.0,       0.30),   # mJ/cm²
        "film_thickness": (_NAN,       0.0),
        "overlay_error":  (5.0,        1.50),   # nm
        "defect_density": (0.02,       0.008),  # defects/cm²
    },
    "Etching": {
        "temperature":    (60.0,       3.0),    # °C  — plasma etch chamber
        "pressure":       (10.0,       0.80),   # mTorr
        "gas_flow":       (100.0,      5.0),    # sccm
        "rf_power":       (500.0,      15.0),   # W
        "exposure_dose":  (_NAN,       0.0),
        "film_thickness": (1000.0,     40.0),   # Å removed
        "overlay_error":  (_NAN,       0.0),
        "defect_density": (0.05,       0.018),
    },
    "Deposition": {
        "temperature":    (400.0,      8.0),    # °C  — CVD/PVD
        "pressure":       (5.0,        0.30),   # mTorr
        "gas_flow":       (200.0,      8.0),    # sccm
        "rf_power":       (300.0,      12.0),   # W
        "exposure_dose":  (_NAN,       0.0),
        "film_thickness": (5000.0,     80.0),   # Å deposited
        "overlay_error":  (_NAN,       0.0),
        "defect_density": (0.03,       0.012),
    },
    "CMP": {
        "temperature":    (25.0,       2.0),    # °C  — polishing pad contact
        "pressure":       (3000.0,     100.0),  # mTorr equivalent for down-force
        "gas_flow":       (_NAN,       0.0),
        "rf_power":       (_NAN,       0.0),
        "exposure_dose":  (_NAN,       0.0),
        "film_thickness": (500.0,      25.0),   # Å removed by CMP
        "overlay_error":  (_NAN,       0.0),
        "defect_density": (0.04,       0.015),
    },
    "Metrology": {
        "temperature":    (23.0,       0.50),   # °C  — measurement tool environment
        "pressure":       (760_000.0,  300.0),  # mTorr — atmospheric
        "gas_flow":       (_NAN,       0.0),
        "rf_power":       (_NAN,       0.0),
        "exposure_dose":  (_NAN,       0.0),
        "film_thickness": (4500.0,     20.0),   # Å measured (post-CMP)
        "overlay_error":  (6.0,        1.0),    # nm measured
        "defect_density": (0.03,       0.010),
    },
}

ROOT_CAUSE_DESCRIPTIONS: dict[str, str] = {
    "normal":      "No anomaly detected",
    "drift":       "Gradual parameter drift - possible consumable wear or supply variation",
    "spike":       "Transient excursion - possible equipment event or measurement outlier",
    "step_shift":  "Sudden level shift - possible calibration event or component replacement",
    "tool_offset": "Systematic tool-to-tool offset - chamber matching recommended",
}

PARAM_COLS: list[str] = [
    "temperature", "pressure", "gas_flow", "rf_power",
    "exposure_dose", "film_thickness", "overlay_error", "defect_density",
]


class SyntheticProcessDataGenerator:
    """Generates a simulated semiconductor process parameter dataset.

    The output mimics the structure of lot/wafer/step time-series data collected
    from a fab MES/SPC system, with configurable anomaly injection.

    All values are drawn from statistical distributions — NOT real equipment data.

    Example:
        gen = SyntheticProcessDataGenerator(seed=42)
        df = gen.generate(n_lots=50, n_wafers_per_lot=25)
        df.to_csv("data/synthetic/process_data.csv", index=False, encoding="utf-8")
    """

    def __init__(self, seed: int = RANDOM_SEED) -> None:
        self._rng = np.random.default_rng(seed)

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(
        self,
        n_lots: int = 50,
        n_wafers_per_lot: int = 25,
        process_steps: list[str] | None = None,
        anomaly_rate: float = 0.05,
        anomaly_types: list[str] | None = None,
    ) -> pd.DataFrame:
        """Generate a synthetic process dataset.

        Args:
            n_lots: Number of lot IDs (e.g. 50 lots ~ 2 months of production).
            n_wafers_per_lot: Wafers per lot (typically 25 in 300mm fabs).
            process_steps: Steps to simulate. Defaults to all five standard steps.
            anomaly_rate: Approximate fraction of rows with injected anomalies.
                          The actual fraction depends on which anomaly types are active.
            anomaly_types: Subset of {"drift","spike","step_shift","tool_offset"}.
                           Defaults to all four types.

        Returns:
            DataFrame with one row per (lot × wafer × step), columns documented in
            docs/data_contract.md.

        Raises:
            ValueError: If an unknown anomaly type is requested.
        """
        steps = process_steps if process_steps is not None else PROCESS_STEPS
        types = anomaly_types if anomaly_types is not None else ["drift", "spike", "step_shift", "tool_offset"]
        self._validate_anomaly_types(types)

        df = self._build_skeleton(n_lots, n_wafers_per_lot, steps)
        df = self._fill_parameters(df, steps)
        df = self._inject_anomalies(df, n_lots, types)
        df = self._compute_yield(df)
        df["suspected_root_cause"] = df["anomaly_type"].map(ROOT_CAUSE_DESCRIPTIONS).astype(object)

        # Drop internal tracking columns before returning
        df = df.drop(columns=["lot_index", "wafer_index", "step_index"])
        df = df.reset_index(drop=True)

        n_anomalies = int(df["anomaly_label"].sum())
        logger.info(
            f"[synthetic] {len(df):,} rows | {n_anomalies} anomalies "
            f"({n_anomalies / len(df):.1%}) | steps={steps}"
        )
        return df

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _validate_anomaly_types(types: list[str]) -> None:
        valid = VALID_ANOMALY_TYPES - {"normal"}
        unknown = set(types) - valid
        if unknown:
            raise ValueError(f"Unknown anomaly type(s): {unknown}. Valid: {sorted(valid)}")

    def _build_skeleton(
        self, n_lots: int, n_wafers_per_lot: int, steps: list[str]
    ) -> pd.DataFrame:
        """Create the (lot, wafer, step) index structure with timestamps."""
        records = []
        base_ts = pd.Timestamp("2024-01-01 00:00:00")
        step_duration = pd.Timedelta(minutes=30)

        for lot_i in range(n_lots):
            lot_id = f"LOT_{lot_i + 1:04d}"
            lot_start = base_ts + lot_i * n_wafers_per_lot * len(steps) * step_duration

            for w_i in range(n_wafers_per_lot):
                wafer_id = f"W_{w_i + 1:03d}"
                wafer_start = lot_start + w_i * len(steps) * step_duration

                for s_i, step in enumerate(steps):
                    records.append(
                        {
                            "lot_id":      lot_id,
                            "lot_index":   lot_i,
                            "wafer_id":    wafer_id,
                            "wafer_index": w_i,
                            "process_step": step,
                            "step_index":  s_i,
                            "timestamp":   wafer_start + s_i * step_duration,
                        }
                    )

        return pd.DataFrame(records)

    def _fill_parameters(self, df: pd.DataFrame, steps: list[str]) -> pd.DataFrame:
        """Draw process parameters from step-specific normal distributions."""
        for col in PARAM_COLS:
            df[col] = _NAN

        for step in steps:
            mask = df["process_step"] == step
            n = int(mask.sum())
            for col, (mean, std) in STEP_NOMINALS[step].items():
                if np.isnan(mean):
                    continue  # parameter not applicable for this step
                vals = df[col].values.copy()
                vals[mask.values] = self._rng.normal(mean, std, size=n)
                df[col] = vals

        # Physical densities cannot be negative — clamp after generation
        df["defect_density"] = df["defect_density"].clip(lower=0.0)

        return df

    def _inject_anomalies(
        self, df: pd.DataFrame, n_lots: int, types: list[str]
    ) -> pd.DataFrame:
        n = len(df)
        lot_idx = df["lot_index"].values
        steps = df["process_step"].values

        flags = np.zeros(n, dtype=bool)
        atypes = np.full(n, "normal", dtype=object)

        # ── Drift: last 20% of lots, temperature gradually increases ──────────
        if "drift" in types:
            drift_start = int(n_lots * 0.80)
            is_drift_lot = lot_idx >= drift_start
            progression = np.where(
                is_drift_lot,
                (lot_idx - drift_start) / max(n_lots - drift_start, 1),
                0.0,
            )
            temp = df["temperature"].values.copy()
            temp += progression * 6.0  # up to +6 °C
            df["temperature"] = temp

            drift_anomaly = is_drift_lot & (progression > 0.30)
            flags |= drift_anomaly
            atypes[drift_anomaly] = "drift"

        # ── Step-shift: lots 70%+, rf_power jumps permanently in Etching ─────
        if "step_shift" in types:
            shift_start = int(n_lots * 0.70)
            is_shift = (lot_idx >= shift_start) & (steps == "Etching")
            rf_std = STEP_NOMINALS["Etching"]["rf_power"][1]
            rf = df["rf_power"].values.copy()
            rf[is_shift] += 2.5 * rf_std
            df["rf_power"] = rf
            flags[is_shift] = True
            _set_atype_if_normal(atypes, is_shift, "step_shift")

        # ── Spike: ~1.5% of rows, temperature spikes to ±5σ ──────────────────
        if "spike" in types:
            n_spikes = max(1, int(n * 0.015))
            spike_idx = self._rng.choice(n, size=n_spikes, replace=False)
            temp = df["temperature"].values.copy()
            spike_steps = steps[spike_idx]
            t_means = np.array([STEP_NOMINALS[s]["temperature"][0] for s in spike_steps])
            t_stds = np.array([STEP_NOMINALS[s]["temperature"][1] for s in spike_steps])
            directions = self._rng.choice([-1.0, 1.0], size=n_spikes)
            temp[spike_idx] = t_means + directions * t_stds * 5.0
            df["temperature"] = temp
            flags[spike_idx] = True
            _set_atype_if_normal(atypes, spike_idx, "spike")

        # ── Tool-offset: first 8% of lots, Deposition temperature biased high ─
        if "tool_offset" in types:
            tool_end = max(1, int(n_lots * 0.08))
            is_tool = (lot_idx < tool_end) & (steps == "Deposition")
            t_std = STEP_NOMINALS["Deposition"]["temperature"][1]
            temp = df["temperature"].values.copy()
            temp[is_tool] += 2.0 * t_std
            df["temperature"] = temp
            flags[is_tool] = True
            _set_atype_if_normal(atypes, is_tool, "tool_offset")

        df["anomaly_label"] = flags
        df["anomaly_type"] = atypes
        return df

    def _compute_yield(self, df: pd.DataFrame) -> pd.DataFrame:
        noise = self._rng.normal(0.0, 0.02, size=len(df))
        penalty = df["anomaly_label"].astype(float).values * self._rng.uniform(
            0.05, 0.20, size=len(df)
        )
        df["yield_rate"] = np.clip(0.95 - penalty + noise, 0.0, 1.0)
        return df


# ── Module-level helper ────────────────────────────────────────────────────────

def _set_atype_if_normal(
    atypes: np.ndarray, mask: np.ndarray | list, label: str
) -> None:
    """Set anomaly type only for rows currently labeled 'normal'."""
    idx = np.where(mask)[0] if isinstance(mask, np.ndarray) and mask.dtype == bool else np.asarray(mask)
    is_normal = atypes[idx] == "normal"
    atypes[idx[is_normal]] = label
