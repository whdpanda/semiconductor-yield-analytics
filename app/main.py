"""Streamlit multi-page application entry point.

Launch: streamlit run app/main.py
"""

import streamlit as st

st.set_page_config(
    page_title="Semiconductor Yield Analytics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Semiconductor Yield & Process Analytics Platform")
st.markdown(
    """
    **A portfolio project demonstrating data engineering and ML techniques
    applied to semiconductor manufacturing analytics.**

    Use the sidebar to navigate between modules.

    | Module | Description |
    |--------|-------------|
    | **Wafer Map Classifier** | CNN-based defect pattern classification on WM-811K dataset |
    | **SPC Dashboard** | Western Electric Rules + ML anomaly detection on process data |

    > *Data sources: WM-811K public dataset & simulated process data.
    > Results do not represent real fab performance.*
    """
)
