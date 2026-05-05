"""Streamlit demo: WaferCNN Wafer Map Defect Classifier.

Launch:
    streamlit run src/semiconductor_yield/dashboard/wafer_demo.py
    -- or --
    python scripts/run_wafer_demo.py

This is a PORTFOLIO DEMO using the public WM-811K dataset.
Do not interpret predictions as real fab production decisions.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the package is importable when run as a standalone Streamlit page
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from semiconductor_yield.config import MODELS_DIR, WAFER_DEFECT_CLASSES
from semiconductor_yield.wafer.demo_samples import DEMO_GENERATORS, generate_demo_sample
from semiconductor_yield.wafer.inference import InferenceResult, WaferInference, parse_wafer_input

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Wafer Map Classifier Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CHECKPOINT = MODELS_DIR / "wafer_cnn_best.pth"
_CLASS_NAMES = list(WAFER_DEFECT_CLASSES)

# ── Cached resources ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model...")
def _load_engine() -> tuple[WaferInference, bool]:
    """Returns (engine, is_demo). Cached for the session lifetime."""
    try:
        engine = WaferInference.from_checkpoint(_CHECKPOINT)
        return engine, False
    except FileNotFoundError:
        return WaferInference.demo(), True


# ── Pure helper functions (testable without Streamlit) ────────────────────────

def make_wafer_figure(
    wmap: np.ndarray,
    title: str = "Wafer Map",
    is_synthetic: bool = False,
) -> plt.Figure:
    """Return a matplotlib Figure visualising a wafer map.

    Args:
        wmap: 2-D array. Values:
              - after preprocessing (model input): floats in [0, 1]
              - raw: integers in {0, 1, 2}
        title: Figure title.
        is_synthetic: If True, adds a 'SYNTHETIC' annotation.

    Returns:
        matplotlib Figure. Caller is responsible for closing it.
    """
    from matplotlib.colors import ListedColormap

    # Support both raw {0,1,2} and normalised {0.0, 0.5, 1.0}
    arr = np.asarray(wmap, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(4, 4), facecolor="#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    # Custom 3-colour map: background=dark, good=green, defect=red
    cmap = ListedColormap(["#2b2b3b", "#4caf50", "#f44336"])
    vmin, vmax = (0.0, 1.0) if arr.max() <= 1.05 else (0.0, 2.0)
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

    ax.set_title(title, color="white", fontsize=10, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])

    if is_synthetic:
        ax.text(
            0.03, 0.97, "SYNTHETIC",
            transform=ax.transAxes,
            color="#ffcc00", fontsize=8, fontweight="bold",
            va="top", ha="left",
            bbox=dict(facecolor="#1e1e2e", alpha=0.7, edgecolor="none", pad=2),
        )

    fig.tight_layout(pad=0.5)
    return fig


def make_prob_figure(
    top_k: list[tuple[str, float]],
    predicted_class: str,
) -> plt.Figure:
    """Return a horizontal bar chart of top-k class probabilities.

    Args:
        top_k: List of (class_name, probability) sorted descending.
        predicted_class: The top-1 predicted class name (highlighted in colour).

    Returns:
        matplotlib Figure.
    """
    names = [t[0] for t in top_k]
    probs = [t[1] for t in top_k]

    fig, ax = plt.subplots(figsize=(5, max(2.4, len(names) * 0.45)), facecolor="#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    colors = ["#4caf50" if n == predicted_class else "#5c7cfa" for n in names]
    bars = ax.barh(range(len(names)), probs, color=colors, height=0.6)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, color="white", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability", color="#aaaaaa", fontsize=8)
    ax.tick_params(axis="x", colors="#aaaaaa", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444")
    ax.spines["bottom"].set_color("#444")

    for bar, prob in zip(bars, probs):
        ax.text(
            min(prob + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
            f"{prob:.1%}",
            va="center", ha="left", color="white", fontsize=8,
        )

    fig.tight_layout(pad=0.5)
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────

def _render_sidebar() -> tuple[np.ndarray | None, bool, str]:
    """Render sidebar and return (wafer_map, is_synthetic, source_label).

    Returns None for wafer_map if no input has been provided yet.
    """
    st.sidebar.header("Input")

    source = st.sidebar.radio(
        "Data source",
        ["Demo sample", "Upload file"],
        index=0,
    )

    wafer_map: np.ndarray | None = None
    is_synthetic = False
    source_label = ""

    if source == "Demo sample":
        class_choice = st.sidebar.selectbox(
            "Defect class",
            options=_CLASS_NAMES,
            index=_CLASS_NAMES.index("Center"),
        )
        seed = st.sidebar.number_input("Seed", value=42, min_value=0, max_value=9999)
        if st.sidebar.button("Generate sample", type="primary"):
            wafer_map = generate_demo_sample(class_choice, seed=int(seed))
            is_synthetic = True
            source_label = f"Demo: {class_choice} (seed={seed})"
            st.session_state["wafer_map"] = wafer_map
            st.session_state["is_synthetic"] = is_synthetic
            st.session_state["source_label"] = source_label
        elif "wafer_map" in st.session_state and st.session_state.get("source_label", "").startswith("Demo"):
            wafer_map = st.session_state["wafer_map"]
            is_synthetic = st.session_state.get("is_synthetic", True)
            source_label = st.session_state.get("source_label", "")

    else:  # Upload file
        uploaded = st.sidebar.file_uploader(
            "Wafer map file",
            type=["npy", "pkl", "csv"],
            help="2-D array. Values should be in {0, 1, 2}: 0=background, 1=good, 2=fail.",
        )
        if uploaded is not None:
            try:
                wafer_map = parse_wafer_input(uploaded.read(), filename=uploaded.name)
                is_synthetic = False
                source_label = f"Upload: {uploaded.name}"
                st.session_state["wafer_map"] = wafer_map
                st.session_state["is_synthetic"] = is_synthetic
                st.session_state["source_label"] = source_label
            except (ValueError, Exception) as exc:
                st.sidebar.error(f"Could not parse file: {exc}")
        elif "wafer_map" in st.session_state and not st.session_state.get("source_label", "").startswith("Demo"):
            wafer_map = st.session_state["wafer_map"]
            is_synthetic = st.session_state.get("is_synthetic", False)
            source_label = st.session_state.get("source_label", "")

    # Model status
    st.sidebar.divider()
    engine, is_demo_engine = _load_engine()
    if is_demo_engine:
        st.sidebar.warning(
            "**Model: Demo mode**\n\n"
            "Checkpoint not found. Predictions use a randomly-initialised "
            "model and are MEANINGLESS.\n\n"
            "Train the model first:\n```\npython scripts/train_wafer_cnn.py\n```"
        )
    else:
        st.sidebar.success(f"**Model: Trained**\n\nCheckpoint: `{_CHECKPOINT.name}`")

    return wafer_map, is_synthetic, source_label


# ── Main page ──────────────────────────────────────────────────────────────────

def _render_prediction(result: InferenceResult) -> None:
    """Render prediction results in the right column."""
    if result.is_demo:
        st.warning(
            "**Demo mode — predictions are meaningless.**  "
            "The model has randomly initialised weights. "
            "Train the model with `python scripts/train_wafer_cnn.py` for real results."
        )

    # Top-line metrics
    conf_pct = f"{result.confidence:.1%}"
    conf_color = (
        "normal" if result.confidence >= 0.60
        else "off" if result.confidence >= 0.30
        else "inverse"
    )
    col_a, col_b = st.columns(2)
    col_a.metric("Predicted Class", result.predicted_class)
    col_b.metric("Confidence", conf_pct)

    st.markdown("**Top-5 class probabilities**")
    prob_fig = make_prob_figure(result.top_k, result.predicted_class)
    st.pyplot(prob_fig, use_container_width=True)
    plt.close(prob_fig)

    with st.expander("All class probabilities"):
        import pandas as pd
        df = pd.DataFrame(result.top_k, columns=["Class", "Probability"])
        df["Probability"] = df["Probability"].map(lambda p: f"{p:.4f}")
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_real_world_notes() -> None:
    st.divider()
    st.subheader("About This Demo")
    st.markdown("""
This is a **portfolio project** demonstrating ML techniques on the public
[WM-811K wafer map dataset](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map).
**It is not a production fab system.**

Real fab deployment of a wafer map classifier would additionally require:

| Consideration | Why it matters |
|---|---|
| **Recipe drift** | Control limits must adapt as process baseline shifts over time |
| **Tool-to-tool variation** | Chamber-to-chamber offsets can dominate the signal |
| **Domain shift** | New product lines, new process nodes require model re-validation or retraining |
| **Model monitoring** | Accuracy can degrade silently; requires periodic backtesting against labelled events |
| **False-call cost** | A wrong classification may trigger unnecessary engineer investigation or tool hold |
| **Regulatory trail** | ISO / SEMI standards require documented, audited decision logic |

See `docs/interview_notes.md` for detailed design considerations.
""")


def main() -> None:
    # ── Page header ────────────────────────────────────────────────────────────
    st.title("Wafer Map Defect Classifier")
    st.caption(
        "Portfolio demo · WM-811K public dataset · "
        "Not a production system · Results do not represent real fab performance"
    )
    st.divider()

    # ── Sidebar ────────────────────────────────────────────────────────────────
    wafer_map, is_synthetic, source_label = _render_sidebar()

    # ── Main layout ────────────────────────────────────────────────────────────
    if wafer_map is None:
        st.info(
            "**No input yet.** \n\n"
            "Use the sidebar to generate a demo sample or upload a wafer map file."
        )
        _render_real_world_notes()
        return

    left, right = st.columns([2, 3], gap="large")

    with left:
        st.subheader("Wafer Map")
        if source_label:
            st.caption(source_label)
        wafer_fig = make_wafer_figure(wafer_map, title="", is_synthetic=is_synthetic)
        st.pyplot(wafer_fig, use_container_width=True)
        plt.close(wafer_fig)
        st.caption(
            "**Legend** — dark: background  |  green: good die  |  red: defective die"
        )

    with right:
        st.subheader("Prediction")
        engine, _ = _load_engine()
        with st.spinner("Running inference..."):
            result = engine.predict(wafer_map, top_k=5)
        _render_prediction(result)

    _render_real_world_notes()


if __name__ == "__main__":
    main()
else:
    # Streamlit runs the file as a module, not __main__, so call main() directly.
    main()
