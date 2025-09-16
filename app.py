import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

from src.data_loader import load_builtin_data, load_and_preprocess_raw
from src.model_geophysics import isolate_decay_period, fit_omori, omori_law
from src.model_econometrics import analyze_residuals_arima, fit_garch
from src.visualization import plot_omori_fit, create_acf_pacf_plot, plot_hybrid_nowcast

st.set_page_config(page_title="Aftershock Nowcaster", layout="wide", initial_sidebar_state="collapsed")

# Inject custom CSS for Enterprise Dark UI
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #C9D1D9;
    }
    .metric-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #58A6FF;
    }
    .metric-label {
        font-size: 12px;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .danger-high { color: #F85149; font-weight: bold; }
    .danger-medium { color: #D29922; font-weight: bold; }
    .danger-low { color: #3FB950; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title(" AFTERSHOCK NOWCASTER")
st.markdown("**Hybrid Geophysical-Econometric Dashboard** | *System Status: OPERATIONAL* 🔸")
st.markdown("---")

tab1, tab2 = st.tabs(["1 LIVE ANALYSIS (Turkey-Syria 2023 Dataset)", "2 UPLOAD & ANALYZE YOUR DATA"])

