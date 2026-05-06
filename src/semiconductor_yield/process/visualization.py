"""SPC control chart generation.

All chart functions use lazy matplotlib imports so the module can be imported
in headless environments (CI, tests) without side-effects. The calling script
or conftest.py is responsible for setting the backend before import.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from semiconductor_yield.process.spc import ControlLimits


def plot_control_chart(
    values: np.ndarray,
    limits: ControlLimits,
    violations_df: pd.DataFrame,
    feature: str,
    group_label: str,
    output_path: Path,
) -> None:
    """Generate and save one SPC control chart PNG.

    Args:
        values: 1-D measurement array (NaN allowed; plotted as gaps).
        limits: Pre-computed control limits for this (feature, group).
        violations_df: Rows from the violations table for this (feature, group).
                       Must contain columns ``series_index`` and ``value``.
        feature: Y-axis label and chart title component.
        group_label: Process step name (used in title).
        output_path: Destination PNG path; parent directory is created if needed.
    """
    import matplotlib.pyplot as plt

    n = len(values)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(14, 4))

    # Subtle zone fills
    ax.fill_between(x, limits.ucl_2, limits.ucl,   alpha=0.07, color="red",        label="_nolegend_")
    ax.fill_between(x, limits.lcl,   limits.lcl_2, alpha=0.07, color="red",        label="_nolegend_")
    ax.fill_between(x, limits.ucl_1, limits.ucl_2, alpha=0.07, color="darkorange",  label="_nolegend_")
    ax.fill_between(x, limits.lcl_2, limits.lcl_1, alpha=0.07, color="darkorange",  label="_nolegend_")

    # Data series
    ax.plot(
        x, values,
        color="steelblue", linewidth=0.8, marker="o", markersize=2.5,
        alpha=0.85, label=feature, zorder=3,
    )

    # Control limit lines
    ax.axhline(limits.mean,  color="forestgreen", linewidth=1.5, linestyle="-",
               label=f"Mean = {limits.mean:.4g}")
    ax.axhline(limits.ucl,   color="red",         linewidth=1.2, linestyle="--",
               label=f"UCL / LCL (±3σ)  {limits.ucl:.4g} / {limits.lcl:.4g}")
    ax.axhline(limits.lcl,   color="red",         linewidth=1.2, linestyle="--", label="_nolegend_")
    ax.axhline(limits.ucl_2, color="darkorange",  linewidth=0.8, linestyle=":",  label="±2σ")
    ax.axhline(limits.lcl_2, color="darkorange",  linewidth=0.8, linestyle=":",  label="_nolegend_")
    ax.axhline(limits.ucl_1, color="goldenrod",   linewidth=0.8, linestyle=":",  label="±1σ")
    ax.axhline(limits.lcl_1, color="goldenrod",   linewidth=0.8, linestyle=":",  label="_nolegend_")

    # Violation markers
    if len(violations_df) > 0 and "series_index" in violations_df.columns:
        vi = violations_df["series_index"].values
        vv = violations_df["value"].values
        ax.scatter(
            vi, vv,
            color="red", zorder=6, s=60, marker="x", linewidths=1.8,
            label=f"{len(violations_df)} SPC signal(s)",
        )

    ax.set_title(
        f"SPC Control Chart  |  {feature}  —  {group_label}"
        "\n[SIMULATED DATA — not real fab measurements]",
        fontsize=10, fontweight="bold", pad=8,
    )
    ax.set_xlabel("Sample Index", fontsize=9)
    ax.set_ylabel(feature, fontsize=9)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.85)
    ax.grid(True, alpha=0.25, linewidth=0.5)

    # Footer note: SPC signals are warnings, Rule 4 context
    n_rule4 = (
        int((violations_df["rule"] == "Rule 4").sum())
        if len(violations_df) > 0 and "rule" in violations_df.columns
        else 0
    )
    footer = (
        "SPC signals are process control warnings, not confirmed root causes.  "
        "Simulated data only."
    )
    if n_rule4 > 0:
        footer += (
            f"  Rule 4 (LOW): {n_rule4} raw signal(s) — sustained drift produces "
            "many consecutive windows; see spc_events.csv for event count."
        )
    ax.text(
        0.0, -0.12, footer,
        transform=ax.transAxes, fontsize=6.0, alpha=0.65,
        verticalalignment="top", color="dimgray", style="italic",
        wrap=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.debug(f"Saved: {output_path}")


def plot_spc_summary(
    df: pd.DataFrame,
    limits_map: dict[tuple[str, str], ControlLimits],
    violations_df: pd.DataFrame,
    output_dir: Path,
    timestamp_col: str = "timestamp",
    group_col: str = "process_step",
) -> dict[str, Path]:
    """Generate one control chart PNG per (feature, process_step) pair.

    Only pairs present in ``limits_map`` are charted (i.e. only groups that
    had enough valid data for limit computation).

    Returns:
        Dict mapping ``"{step}__{feature}"`` → output Path for every chart saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    created: dict[str, Path] = {}

    for (feature, step), limits in sorted(limits_map.items()):
        mask = df[group_col] == step
        group_df = df[mask].sort_values(timestamp_col).reset_index(drop=True)
        values = group_df[feature].values.astype(float)

        if len(violations_df) > 0:
            v_mask = (
                (violations_df["feature"] == feature)
                & (violations_df["process_step"] == step)
            )
            step_viols = violations_df[v_mask].reset_index(drop=True)
        else:
            step_viols = pd.DataFrame(columns=["series_index", "value"])

        safe_feature = feature.replace(" ", "_")
        safe_step = step.replace(" ", "_")
        out_path = output_dir / f"{safe_step}__{safe_feature}.png"

        plot_control_chart(
            values=values,
            limits=limits,
            violations_df=step_viols,
            feature=feature,
            group_label=step,
            output_path=out_path,
        )
        created[f"{step}__{feature}"] = out_path

    logger.info(f"[visualization] {len(created)} control charts saved → {output_dir}")
    return created
