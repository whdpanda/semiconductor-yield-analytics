"""Unified Streamlit dashboard: Semiconductor Yield & Process Analytics.

Launch:
    streamlit run src/semiconductor_yield/dashboard/app.py
    -- or --
    python scripts/run_dashboard.py

Portfolio project -- public WM-811K dataset + synthetic process data.
Not a production fab system.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from semiconductor_yield.config import (
    MODELS_DIR,
    PROCESS_REPORTS_DIR,
    RANDOM_SEED,
    SYNTHETIC_DIR,
    WAFER_DEFECT_CLASSES,
)
from semiconductor_yield.dashboard.data_helpers import (
    format_rca_candidates,
    get_anomaly_rate_by_step,
    get_available_charts,
    get_violation_summary,
    load_anomaly_scores,
    load_anomaly_summary,
    load_process_data,
    load_spc_violations,
)
from semiconductor_yield.wafer.demo_samples import generate_demo_sample
from semiconductor_yield.wafer.inference import InferenceResult, WaferInference, parse_wafer_input

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Semiconductor Yield Analytics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── File paths ─────────────────────────────────────────────────────────────────
_SPC_CSV       = PROCESS_REPORTS_DIR / "spc_violations.csv"
_SCORES_CSV    = PROCESS_REPORTS_DIR / "anomaly_scores.csv"
_SUMMARY_JSON  = PROCESS_REPORTS_DIR / "anomaly_summary.json"
_PROCESS_CSV   = SYNTHETIC_DIR / "process_data.csv"
_CHARTS_DIR    = PROCESS_REPORTS_DIR / "charts"
_CHECKPOINT    = MODELS_DIR / "wafer_cnn_best.pth"


# ── Streamlit-cached data loaders ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _spc_violations() -> pd.DataFrame | None:
    return load_spc_violations(_SPC_CSV)

@st.cache_data(show_spinner=False)
def _anomaly_scores() -> pd.DataFrame | None:
    return load_anomaly_scores(_SCORES_CSV)

@st.cache_data(show_spinner=False)
def _anomaly_summary() -> dict | None:
    return load_anomaly_summary(_SUMMARY_JSON)

@st.cache_data(show_spinner=False)
def _process_data() -> pd.DataFrame | None:
    return load_process_data(_PROCESS_CSV)

@st.cache_resource(show_spinner="Loading model ...")
def _wafer_engine() -> tuple[WaferInference, bool]:
    """Returns (engine, is_demo). Cached for session lifetime."""
    try:
        return WaferInference.from_checkpoint(_CHECKPOINT), False
    except FileNotFoundError:
        return WaferInference.demo(), True


# ── Figure builders ────────────────────────────────────────────────────────────

def _wafer_figure(
    wmap: np.ndarray,
    is_synthetic: bool = False,
    title: str = "",
) -> plt.Figure:
    from matplotlib.colors import ListedColormap
    arr = np.asarray(wmap, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="#1e1e2e")
    ax.set_facecolor("#1e1e2e")
    cmap = ListedColormap(["#2b2b3b", "#4caf50", "#f44336"])
    vmin, vmax = (0.0, 1.0) if arr.max() <= 1.05 else (0.0, 2.0)
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, color="white", fontsize=9, pad=4)
    if is_synthetic:
        ax.text(0.03, 0.97, "SYNTHETIC", transform=ax.transAxes,
                color="#ffcc00", fontsize=7, fontweight="bold",
                va="top", ha="left",
                bbox=dict(facecolor="#1e1e2e", alpha=0.7, edgecolor="none", pad=2))
    fig.tight_layout(pad=0.3)
    return fig


def _prob_figure(
    top_k: list[tuple[str, float]],
    predicted_class: str,
) -> plt.Figure:
    names = [t[0] for t in top_k]
    probs = [t[1] for t in top_k]
    fig, ax = plt.subplots(figsize=(5, max(2.2, len(names) * 0.42)),
                           facecolor="#1e1e2e")
    ax.set_facecolor("#1e1e2e")
    colors = ["#4caf50" if n == predicted_class else "#5c7cfa" for n in names]
    bars = ax.barh(range(len(names)), probs, color=colors, height=0.55)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, color="white", fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Probability", color="#888", fontsize=7)
    ax.tick_params(axis="x", colors="#888", labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#444")
    for bar, prob in zip(bars, probs):
        ax.text(min(prob + 0.02, 0.98), bar.get_y() + bar.get_height() / 2,
                f"{prob:.1%}", va="center", ha="left", color="white", fontsize=7)
    fig.tight_layout(pad=0.4)
    return fig


def _anomaly_trend_figure(
    scores: pd.DataFrame,
    step: str | None = None,
) -> plt.Figure:
    df = scores.copy()
    if step and "process_step" in df.columns:
        df = df[df["process_step"] == step]
    df = df.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 3), facecolor="#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    if "if_score" in df.columns:
        ax.plot(df.index, df["if_score"], color="#5c7cfa", lw=1.1,
                label="Isolation Forest score", alpha=0.85)
    if "ae_score" in df.columns:
        ax.plot(df.index, df["ae_score"], color="#ff7f50", lw=1.1,
                label="Autoencoder recon. error", alpha=0.80)

    flag_col = "ensemble_any" if "ensemble_any" in df.columns else None
    if flag_col and "if_score" in df.columns:
        flagged = df[df[flag_col] == 1]
        ax.scatter(flagged.index, flagged["if_score"],
                   color="#f44336", s=22, zorder=5, label="Flagged (ensemble)")

    ax.set_xlabel("Sample index", color="#aaa", fontsize=8)
    ax.set_ylabel("Score", color="#aaa", fontsize=8)
    ax.tick_params(colors="#aaa", labelsize=7)
    ax.legend(facecolor="#2b2b3b", labelcolor="white", fontsize=7, framealpha=0.9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#444")
    fig.tight_layout(pad=0.4)
    return fig


# ── Reusable disclaimer widget ─────────────────────────────────────────────────

def _data_disclaimer() -> None:
    st.caption(
        "Portfolio demo · WM-811K public dataset + synthetic process data · "
        "Not a production system · Results do not represent real fab performance"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Pages
# ══════════════════════════════════════════════════════════════════════════════

def page_home() -> None:
    st.title("Semiconductor Yield & Process Analytics")
    _data_disclaimer()

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.subheader("Module A — Wafer Map Classification")
        st.markdown("""
- **Model:** WaferCNN, 3 conv blocks + Global Average Pooling, ~94K params
- **Dataset:** WM-811K public dataset, 9 defect pattern classes
- **Class imbalance:** ~79% "none" class → WeightedRandomSampler + macro F1
- **Demo mode:** works without WM-811K data (synthetic patterns)
        """)

    with col_b:
        st.subheader("Module B — Process SPC & Anomaly Detection")
        st.markdown("""
- **SPC:** 4 Western Electric Rules (ISO 7870 / SEMI E10 compatible)
- **ML:** Isolation Forest + Autoencoder (multivariate, unsupervised)
- **Split:** lot-level grouped — no lot_id leakage between train and test
- **RCA:** candidate analysis ranked by SPC + anomaly signal synergy
        """)

    st.divider()

    with st.expander("Data disclosure", expanded=True):
        st.info("""
**All results on this dashboard use non-production data:**

- **Module A:** WM-811K public dataset (~172k labelled wafer maps). Reported metrics \
reflect dataset properties only — not real fab yield performance.
- **Module B:** Synthetically generated process data with controlled anomaly injection \
(5,000 samples, 50 lots, 5 process steps). Results validate pipeline logic, not real \
process performance.

This project has not been deployed or validated in a real semiconductor facility.
        """)

    st.divider()
    st.subheader("Page Guide")
    st.markdown("""
| Page | What you will find |
|------|--------------------|
| **Wafer Map Classification** | Generate or upload a wafer map → defect pattern prediction |
| **Process SPC** | Control charts and Western Electric Rule violation summary |
| **Anomaly Detection** | IF + Autoencoder score trend, flagged sample table, model metrics |
| **RCA Analysis** | Ranked root-cause candidates with evidence and recommended checks |
| **Real Fab Considerations** | Engineering gaps vs. a production-grade deployment |

**Interview demo order (recommended):**
1. Home → data context and disclaimer
2. Wafer Map Classification → live prediction on demo sample
3. Process SPC → control chart for Etching | temperature
4. Anomaly Detection → score trend + flagged table
5. RCA Analysis → top candidate with evidence
6. Real Fab Considerations → proactively raise deployment gaps
    """)


# ── Wafer Map Classification ───────────────────────────────────────────────────

def page_wafer() -> None:
    st.header("Wafer Map Defect Classification")
    _data_disclaimer()

    engine, is_demo = _wafer_engine()

    if is_demo:
        st.warning(
            f"**Demo mode** — checkpoint not found at `{_CHECKPOINT}`.\n\n"
            "Predictions use randomly initialised weights and are **meaningless**. "
            "Train first: `python scripts/train_wafer_cnn.py`"
        )
    else:
        st.success(f"Trained model loaded · `{_CHECKPOINT.name}`")

    ctrl, vis, pred = st.columns([1, 2, 2], gap="medium")

    with ctrl:
        st.subheader("Input")
        source = st.radio("Source", ["Demo sample", "Upload file"],
                          label_visibility="collapsed")

        if source == "Demo sample":
            cls = st.selectbox("Defect class", list(WAFER_DEFECT_CLASSES))
            seed = st.number_input("Seed", value=42, min_value=0, max_value=9999)
            if st.button("Generate", type="primary"):
                wm = generate_demo_sample(cls, seed=int(seed))
                st.session_state.update(
                    w_map=wm, w_synthetic=True,
                    w_label=f"{cls} (seed={int(seed)})"
                )
        else:
            f = st.file_uploader("Upload .npy / .pkl / .csv",
                                 type=["npy", "pkl", "csv"])
            if f is not None:
                try:
                    wm = parse_wafer_input(f.read(), filename=f.name)
                    st.session_state.update(
                        w_map=wm, w_synthetic=False, w_label=f.name
                    )
                except ValueError as exc:
                    st.error(str(exc))

    wmap = st.session_state.get("w_map")
    if wmap is None:
        with vis:
            st.info("Select a demo sample or upload a file.")
        return

    result = engine.predict(wmap, top_k=5)

    with vis:
        st.subheader("Wafer Map")
        st.caption(st.session_state.get("w_label", ""))
        fig = _wafer_figure(wmap, is_synthetic=st.session_state.get("w_synthetic", False))
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("dark = background  |  green = good die  |  red = defective die")

    with pred:
        st.subheader("Prediction")
        if result.is_demo:
            st.warning("Untrained model — predictions are meaningless (demo mode)")
        c1, c2 = st.columns(2)
        c1.metric("Predicted class", result.predicted_class)
        c2.metric("Confidence", f"{result.confidence:.1%}")
        st.markdown("**Top-5 probabilities**")
        pf = _prob_figure(result.top_k, result.predicted_class)
        st.pyplot(pf, use_container_width=True)
        plt.close(pf)


# ── Process SPC ────────────────────────────────────────────────────────────────

def page_spc() -> None:
    st.header("Statistical Process Control")
    _data_disclaimer()

    violations = _spc_violations()
    charts = get_available_charts(_CHARTS_DIR)

    if violations is None and not charts:
        st.warning(
            "SPC results not found. Generate them:\n"
            "```\npython scripts/run_spc_analysis.py\n```"
        )
        return

    ctrl, main = st.columns([1, 3], gap="medium")

    with ctrl:
        st.subheader("Filter")
        chart_opts = ["(all)"] + sorted(charts.keys())
        chart_sel = st.selectbox("Step | Feature", options=chart_opts)

        if violations is not None and "severity" in violations.columns:
            sev_opts = ["ALL"] + sorted(violations["severity"].unique().tolist())
            sev_sel = st.selectbox("Severity", sev_opts)
        else:
            sev_sel = "ALL"

    with main:
        # Control chart PNG
        if chart_sel != "(all)" and chart_sel in charts:
            st.subheader(f"Control Chart — {chart_sel}")
            st.image(str(charts[chart_sel]), use_column_width=True)
        elif charts:
            st.info("Select a step | feature combination to display its control chart.")

        if violations is not None and not violations.empty:
            st.subheader("Violations")

            df_show = violations.copy()
            if chart_sel != "(all)" and "|" in chart_sel:
                step_f, feat_f = [s.strip() for s in chart_sel.split("|", 1)]
                df_show = df_show[
                    (df_show["process_step"] == step_f) &
                    (df_show["feature"] == feat_f)
                ]
            if sev_sel != "ALL" and "severity" in df_show.columns:
                df_show = df_show[df_show["severity"] == sev_sel]

            # Summary metrics
            total_v = len(df_show)
            high_v  = int((df_show["severity"] == "HIGH").sum()) if "severity" in df_show.columns else "—"
            steps_v = df_show["process_step"].nunique() if "process_step" in df_show.columns else "—"
            m1, m2, m3 = st.columns(3)
            m1.metric("Violations shown", f"{total_v:,}")
            m2.metric("HIGH severity",    f"{high_v:,}")
            m3.metric("Steps affected",   steps_v)

            tab1, tab2 = st.tabs(["By step & rule", "Raw violations"])
            with tab1:
                summ = get_violation_summary(df_show)
                if not summ.empty:
                    st.dataframe(summ, use_container_width=True, hide_index=True)
            with tab2:
                disp_cols = [c for c in
                             ["timestamp", "process_step", "feature",
                              "rule", "severity", "value", "rule_description"]
                             if c in df_show.columns]
                st.dataframe(
                    df_show[disp_cols].sort_values("timestamp").head(200),
                    use_container_width=True, hide_index=True,
                )
                if len(df_show) > 200:
                    st.caption(f"Showing first 200 of {len(df_show):,} rows")


# ── Anomaly Detection ──────────────────────────────────────────────────────────

def page_anomaly() -> None:
    st.header("Anomaly Detection")
    _data_disclaimer()
    st.caption(
        "Isolation Forest + Autoencoder · Unsupervised · "
        "Evaluated on held-out test split only (8 lots, 1,000 samples)"
    )

    scores  = _anomaly_scores()
    summary = _anomaly_summary()

    if scores is None:
        st.warning(
            "Anomaly scores not found. Run:\n"
            "```\npython scripts/train_process_anomaly.py\n"
            "python scripts/evaluate_process_anomaly.py\n```"
        )
        return

    # Model performance bar
    if summary and "models" in summary:
        mods = summary["models"]
        st.subheader("Model Performance — test split")
        mc = st.columns(4)
        for col, (name, label) in zip(mc, [
            ("isolation_forest", "IF F1"),
            ("autoencoder",      "AE F1"),
            ("ensemble_any",     "Ensemble (any) F1"),
            ("ensemble_all",     "Ensemble (all) F1"),
        ]):
            val = mods.get(name, {}).get("f1", None)
            col.metric(label, f"{val:.3f}" if val is not None else "—")

        with st.expander("Full metrics table"):
            rows = []
            for key, label in [
                ("isolation_forest", "Isolation Forest"),
                ("autoencoder",      "Autoencoder"),
                ("ensemble_any",     "Ensemble (any)"),
                ("ensemble_all",     "Ensemble (all)"),
            ]:
                m = mods.get(key, {})
                rows.append({
                    "Model":      label,
                    "Precision":  f"{m.get('precision', 0):.3f}",
                    "Recall":     f"{m.get('recall', 0):.3f}",
                    "F1":         f"{m.get('f1', 0):.3f}",
                    "ROC-AUC":    f"{m.get('roc_auc', 0):.3f}",
                    "FAR":        f"{m.get('false_alarm_rate', 0):.3f}",
                    "Flagged":    m.get("n_flagged", "—"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(
                "_Moderate recall reflects subtle slow-drift anomalies (+6 °C over final "
                "20% of lots) that both models partially miss — a realistic fab challenge._"
            )

    st.divider()

    # Step filter
    steps = ["(all)"]
    if "process_step" in scores.columns:
        steps += sorted(scores["process_step"].unique().tolist())
    step_sel = st.selectbox("Filter by process step", steps)
    step_val = None if step_sel == "(all)" else step_sel

    # Score trend
    st.subheader("Anomaly Score Trend")
    tf = _anomaly_trend_figure(scores, step=step_val)
    st.pyplot(tf, use_container_width=True)
    plt.close(tf)
    st.caption("Red dots = flagged by ensemble (any). Isolation Forest: higher score → more anomalous. "
               "Autoencoder: reconstruction error, same direction.")

    # Rate table + flagged samples
    col_rate, col_tbl = st.columns([2, 3], gap="medium")

    with col_rate:
        st.subheader("Anomaly Rate by Step")
        rate_df = get_anomaly_rate_by_step(scores)
        if not rate_df.empty:
            st.dataframe(
                rate_df.rename(columns={"rate_pct": "Rate (%)"}),
                use_container_width=True, hide_index=True,
            )

    with col_tbl:
        st.subheader("Flagged Samples")
        flag_col = "ensemble_any" if "ensemble_any" in scores.columns else None
        if flag_col:
            flagged = scores[scores[flag_col] == 1].copy()
            if step_val and "process_step" in flagged.columns:
                flagged = flagged[flagged["process_step"] == step_val]
            disp = [c for c in
                    ["lot_id", "wafer_id", "process_step",
                     "anomaly_type", "if_score", "ae_score"]
                    if c in flagged.columns]
            st.dataframe(
                flagged[disp].head(50).reset_index(drop=True),
                use_container_width=True, hide_index=True,
            )
            if len(flagged) > 50:
                st.caption(f"Showing 50 of {len(flagged)} flagged samples")


# ── RCA Analysis ───────────────────────────────────────────────────────────────

def page_rca() -> None:
    st.header("RCA-style Candidate Analysis")
    _data_disclaimer()

    st.info(
        "This module combines SPC violations and anomaly detector signals to rank "
        "**suspected process steps** as investigation candidates. "
        "All outputs use *candidate* and *suspected* language deliberately — "
        "confirming a root cause requires engineer review of FDC logs, recipe history, "
        "consumable records, and tool maintenance data."
    )

    process_df = _process_data()
    violations = _spc_violations()
    scores     = _anomaly_scores()

    if process_df is None:
        st.warning(
            "Synthetic process data not found. Generate it:\n"
            "```\npython scripts/generate_synthetic_data.py\n```"
        )
        return

    try:
        from semiconductor_yield.process.rca import analyze
        with st.spinner("Running candidate analysis ..."):
            candidates = analyze(
                df=process_df,
                spc_violations=violations,
                anomaly_scores=scores,
                data_source="synthetic",
                top_n=5,
            )
    except Exception as exc:
        st.error(f"RCA analysis failed: {exc}")
        return

    if not candidates:
        st.info("No candidates generated — check that process data includes violations.")
        return

    # Summary table
    st.subheader(f"Top {len(candidates)} Candidates")
    summ = format_rca_candidates(candidates)
    st.dataframe(pd.DataFrame(summ), use_container_width=True, hide_index=True)
    st.divider()

    # Per-candidate detail
    _CONF_LABEL = {"high": "[HIGH]", "medium": "[MEDIUM]", "low": "[LOW]"}
    for i, cand in enumerate(candidates):
        label = _CONF_LABEL.get(cand.confidence_level, "[?]")
        with st.expander(
            f"#{i+1}  {cand.suspected_process_step}  {label}",
            expanded=(i == 0),
        ):
            col_ev, col_chk = st.columns(2)
            with col_ev:
                st.markdown("**Suspicious features**")
                for feat in cand.suspicious_features:
                    st.markdown(f"- `{feat}`")
                st.markdown("**Evidence**")
                for ev in cand.evidence:
                    st.markdown(f"- {ev}")
            with col_chk:
                st.markdown("**Recommended checks**")
                for chk in cand.recommended_checks:
                    st.markdown(f"- {chk}")
                st.markdown(f"**Confidence:** `{cand.confidence_level.upper()}`")
            st.caption(f"_{cand.limitation_note}_")


# ── Real Fab Considerations ────────────────────────────────────────────────────

_CONSIDERATIONS = [
    (
        "Recipe Drift",
        """
Process parameters drift as equipment ages, consumables wear, and recipes are modified.
A model trained on a fixed historical window will degrade silently over time.

**In this project:** Training uses a fixed synthetic dataset; no runtime drift detection.

**In production:**
- Rolling-window SPC baseline recalculation as process conditions evolve
- Input-feature drift monitoring (PSI, KS-test) to trigger retraining alerts
- CUSUM charts for gradual, sustained shifts that WE Rules may miss
""",
    ),
    (
        "Tool-to-Tool Variation",
        """
Chamber-to-chamber offsets within a fleet of nominally identical tools can dominate the
signal and cause systematic false alarms or blind spots for specific tools.

**In this project:** Synthetic data treats all tools as identical; no per-tool structure.

**In production:**
- Per-tool normalization before SPC and ML scoring
- Mixed-effects models separating tool offset from true process variation
- Fleet-level aggregation dashboards to spot systematic outliers
""",
    ),
    (
        "Class Imbalance (Wafer Classification)",
        """
WM-811K has ~79% "none" class. Accuracy is meaningless here — a classifier predicting
"none" for every sample achieves 79% without learning anything.

**In this project:** WeightedRandomSampler re-balances gradient signal; primary metric is
macro F1. Per-class F1 and recall are the production-relevant numbers.

**In production:** Cost-aware threshold calibration per class. Missing an Edge-Ring fault
(systematic equipment issue) is typically far more expensive than a false Loc alarm.
""",
    ),
    (
        "Domain Shift",
        """
New product nodes, process changes, or new equipment generations create data distributions
that differ from the training set, causing silent accuracy degradation.

**In this project:** Training and evaluation share the same data distribution; no drift
simulation.

**In production:**
- Confidence thresholding to route low-confidence predictions to human review
- Periodic re-validation against labelled fault events from recent production
- Domain adaptation or incremental fine-tuning for new product introductions
""",
    ),
    (
        "False Alarm Cost",
        """
In a fab, triggering a false anomaly alarm can initiate a tool hold costing $5k–$10k/hour.
The detection threshold is an engineering decision, not a data science decision.

**In this project:** The ensemble offers `any` (high recall) and `all` (high precision)
voting. Current test-split results: Ensemble(any) F1=0.317, FAR=8.1%;
Ensemble(all) F1=0.130, FAR=1.5%.

**In production:**
- Cost-aware threshold tuning against a labelled holdout with explicit downtime cost model
- Escalation tiers: alert → investigation ticket → tool hold, each requiring increasing
  confidence and engineer sign-off
""",
    ),
    (
        "Model Monitoring & Human-in-the-Loop Review",
        """
ML models in safety-relevant manufacturing contexts require continuous monitoring and
cannot operate as autonomous decision-makers under ISO/SEMI standards.

**In this project:** No runtime monitoring; all outputs carry explicit disclaimers.

**In production:**
- Reconstruction error distribution tracking to detect Autoencoder degradation
- Periodic backtesting against labelled fault events (monthly or per tool PM cycle)
- Full audit trail — every flagged event, threshold used, and engineer action logged
- Credentialed process engineer required before any tool hold action
- SEMI E10 / ISO 7870 compliance documentation for regulatory audit readiness
""",
    ),
]


def page_considerations() -> None:
    st.header("Real Fab Deployment Considerations")
    st.caption(
        "Engineering gaps between this portfolio demo and a production-grade system"
    )
    st.markdown("""
This page documents what a real deployment would require beyond this demo.
Raising these points proactively in an interview demonstrates manufacturing domain
awareness — not just ML proficiency.
""")

    for title, body in _CONSIDERATIONS:
        with st.expander(title):
            st.markdown(body)

    st.divider()
    st.markdown(
        "Reference: `docs/interview_notes.md` — SPC rationale, RCA language constraints, "
        "feature leakage taxonomy, SECOM anonymous mode."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Navigation
# ══════════════════════════════════════════════════════════════════════════════

_PAGES: dict[str, object] = {
    "Home":                        page_home,
    "Wafer Map Classification":    page_wafer,
    "Process SPC":                 page_spc,
    "Anomaly Detection":           page_anomaly,
    "RCA Analysis":                page_rca,
    "Real Fab Considerations":     page_considerations,
}


def main() -> None:
    st.sidebar.title("Navigation")
    page_name = st.sidebar.radio(
        "page",
        options=list(_PAGES.keys()),
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    st.sidebar.caption(
        "**Semiconductor Yield Analytics**\n\n"
        "Portfolio project · Public data only\n\n"
        "Not a production system"
    )
    _PAGES[page_name]()  # type: ignore[operator]


# Streamlit runs this file as a module (not __main__), so call main() directly.
main()
