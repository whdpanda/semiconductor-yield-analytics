"""EDA utilities for the WM-811K wafer map dataset (Module A).

Generates class distribution reports and representative sample visualisations.
All outputs are saved to outputs/reports/wafer/ by default.

Plotting functions use lazy matplotlib imports so they work in headless
environments without requiring an explicit backend to be set at import time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from semiconductor_yield.config import REPORTS_DIR, WAFER_DEFECT_CLASSES
from semiconductor_yield.wafer.data_loader import WaferSample
from semiconductor_yield.wafer.preprocess import imbalance_stats

WAFER_REPORTS_DIR: Path = REPORTS_DIR / "wafer"

# ── Statistics ─────────────────────────────────────────────────────────────────


def class_distribution(samples: list[WaferSample]) -> pd.Series:
    """Return class counts as a Series indexed by class name, sorted descending."""
    counts: dict[str, int] = {c: 0 for c in WAFER_DEFECT_CLASSES}
    for s in samples:
        if s.label_name in counts:
            counts[s.label_name] += 1
    series = pd.Series(counts, dtype=int)
    return series.sort_values(ascending=False)


def wafer_map_size_distribution(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Summarise the (height, width) distribution of raw wafer maps.

    Args:
        df_raw: The raw DataFrame returned by WM811KLoader.load_raw().

    Returns:
        DataFrame with columns: height, width, count, percentage.
        Sorted descending by count.
    """

    def _shape(m: object) -> tuple[int, int]:
        arr = np.array(m)
        if arr.ndim >= 2:
            return int(arr.shape[0]), int(arr.shape[1])
        return 0, 0

    shapes = df_raw["waferMap"].apply(_shape)
    shape_counts = shapes.value_counts()
    total = len(shapes)
    rows = [
        {
            "height":     h,
            "width":      w,
            "count":      int(cnt),
            "percentage": round(cnt / total * 100, 2),
        }
        for (h, w), cnt in shape_counts.items()
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


# ── Visualisation ──────────────────────────────────────────────────────────────


def plot_class_distribution(
    dist: pd.Series,
    output_path: Path,
    title: str = "WM-811K Wafer Defect Class Distribution (labeled wafers only)",
) -> None:
    """Save a horizontal bar chart of class counts with percentage annotations."""
    import matplotlib.pyplot as plt  # lazy import

    fig, ax = plt.subplots(figsize=(10, 5))
    n_classes = len(dist)
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, n_classes))

    bars = ax.barh(range(n_classes), dist.values, color=colors, edgecolor="white")
    total = dist.values.sum()

    for bar, count in zip(bars, dist.values):
        pct = count / total * 100
        ax.text(
            bar.get_width() + total * 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}  ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(dist.index.tolist(), fontsize=10)
    ax.set_xlabel("Sample count", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlim(0, dist.max() * 1.28)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.5, 0.0,
        "Source: WM-811K public dataset (Kaggle). Labeled subset ~172k / 811k wafers.",
        ha="center", fontsize=7, color="#888888",
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved class distribution chart → {output_path}")


def plot_sample_wafer_maps(
    samples: list[WaferSample],
    output_path: Path,
    n_per_class: int = 3,
    title: str = "WM-811K Sample Wafer Maps by Defect Class",
) -> None:
    """Save a grid of example wafer maps — one row per defect class.

    Cells with no available sample (class under-represented in the provided
    sample list) are displayed as a grey placeholder.

    Args:
        samples: WaferSample list; can be a subset of the full dataset.
        output_path: Where to save the PNG.
        n_per_class: Columns in the grid (samples per class).
        title: Figure super-title.
    """
    import matplotlib.pyplot as plt  # lazy import

    # Collect up to n_per_class examples per class
    class_samples: dict[str, list[WaferSample]] = {c: [] for c in WAFER_DEFECT_CLASSES}
    for s in samples:
        bucket = class_samples.get(s.label_name)
        if bucket is not None and len(bucket) < n_per_class:
            bucket.append(s)

    n_classes = len(WAFER_DEFECT_CLASSES)
    fig, axes = plt.subplots(
        n_classes, n_per_class,
        figsize=(n_per_class * 2.3, n_classes * 2.1),
    )
    # Ensure axes is always 2-D even if n_per_class == 1
    if n_per_class == 1:
        axes = axes[:, np.newaxis]

    fig.subplots_adjust(left=0.15, hspace=0.06, wspace=0.05)
    cmap = plt.cm.RdYlGn  # 0=background (white/gray), 1=good die (green), 2=defect (red)

    for row_i, class_name in enumerate(WAFER_DEFECT_CLASSES):
        samps = class_samples[class_name]
        n_avail = len(samps)

        for col_i in range(n_per_class):
            ax = axes[row_i][col_i]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if col_i < n_avail:
                ax.imshow(
                    samps[col_i].wafer_map,
                    cmap=cmap, vmin=0, vmax=2, interpolation="nearest",
                )
            else:
                ax.set_facecolor("#f0f0f0")
                if col_i == 0:  # only show "no data" in first empty cell per row
                    ax.text(
                        0.5, 0.5, "—",
                        ha="center", va="center",
                        transform=ax.transAxes, fontsize=14, color="#aaaaaa",
                    )

            # Row label on first column (using axes-fraction coordinates)
            if col_i == 0:
                ax.text(
                    -0.08, 0.60,
                    class_name,
                    ha="right", va="center",
                    transform=ax.transAxes,
                    fontsize=8.5, fontweight="bold",
                )
                ax.text(
                    -0.08, 0.38,
                    f"n={n_avail:,}",
                    ha="right", va="center",
                    transform=ax.transAxes,
                    fontsize=7, color="#666666",
                )

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    fig.text(
        0.5, -0.005,
        "Values: 0 = background, 1 = good die, 2 = defect die  |  Source: WM-811K (Kaggle)",
        ha="center", fontsize=7, color="#888888",
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved sample wafer maps → {output_path}")


# ── Orchestrator ───────────────────────────────────────────────────────────────


def run_eda(
    samples: list[WaferSample],
    df_raw: pd.DataFrame | None = None,
    output_dir: Path = WAFER_REPORTS_DIR,
) -> dict[str, Path]:
    """Run the full EDA pipeline and save all artefacts to output_dir.

    Args:
        samples: Labeled WaferSample list from WM811KLoader.load(labeled_only=True).
        df_raw: Raw DataFrame for size-distribution analysis. Optional.
        output_dir: Destination directory (created if missing).

    Returns:
        Dict mapping output key → absolute Path of the saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    # 1 — Class distribution CSV
    dist = class_distribution(samples)
    csv_path = output_dir / "class_distribution.csv"
    pd.DataFrame({"class_name": dist.index, "count": dist.values}).to_csv(
        csv_path, index=False
    )
    outputs["class_distribution_csv"] = csv_path
    logger.info(f"Saved class distribution CSV → {csv_path}")

    # 2 — Class imbalance report
    stats_df = imbalance_stats(samples)
    report_path = output_dir / "imbalance_report.csv"
    stats_df.to_csv(report_path, index=False)
    outputs["imbalance_report"] = report_path
    logger.info(f"Saved imbalance report → {report_path}")

    # 3 — Class distribution bar chart
    plot_path = output_dir / "class_distribution.png"
    plot_class_distribution(dist, plot_path)
    outputs["class_distribution_png"] = plot_path

    # 4 — Sample wafer maps grid
    maps_path = output_dir / "sample_wafer_maps.png"
    plot_sample_wafer_maps(samples, maps_path)
    outputs["sample_wafer_maps_png"] = maps_path

    # 5 — Wafer map size distribution (only if raw DataFrame provided)
    if df_raw is not None:
        size_df = wafer_map_size_distribution(df_raw)
        size_path = output_dir / "wafer_map_sizes.csv"
        size_df.to_csv(size_path, index=False)
        outputs["size_distribution_csv"] = size_path
        logger.info(f"Saved size distribution → {size_path}")

    logger.info(f"EDA complete — {len(outputs)} artefacts saved to {output_dir}")
    return outputs
