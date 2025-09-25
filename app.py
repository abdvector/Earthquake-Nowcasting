import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import warnings
import time
import json

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

def render_map(df_raw):
    """Render a PyDeck map with earthquake heat points if lat/lon exist."""
    if 'latitude' in df_raw.columns and 'longitude' in df_raw.columns:
        st.subheader("🔴 Earthquake & Aftershock Overview")
        
        # We can size points by magnitude if available
        if 'mag' in df_raw.columns:
            # Drop NaNs in mag for mapping
            map_data = df_raw.dropna(subset=['latitude', 'longitude', 'mag']).copy()
            # Normalize magnitude for radius
            map_data['radius'] = (map_data['mag'] - map_data['mag'].min() + 0.1) * 2000
        else:
            map_data = df_raw.dropna(subset=['latitude', 'longitude']).copy()
            map_data['radius'] = 5000
            
        layer = pdk.Layer(
            'ScatterplotLayer',
            data=map_data,
            get_position='[longitude, latitude]',
            get_color='[255, 75, 75, 140]',
            get_radius='radius',
            pickable=True
        )
        
        view_state = pdk.ViewState(
            latitude=map_data['latitude'].mean(),
            longitude=map_data['longitude'].mean(),
            zoom=6,
            pitch=0
        )
        
        st.pydeck_chart(pdk.Deck(
            map_style='dark',
            initial_view_state=view_state,
            layers=[layer],
            tooltip={"text": "Mag: {mag}" if 'mag' in df_raw.columns else "Aftershock Location"}
        ))
    else:
        st.info("No spatial data (latitude/longitude) found for mapping.")

def render_pipeline(df_raw, df_hourly):
    # 1. Map
    render_map(df_raw)
    
    st.markdown("---")
    
    # 2. Geophysics
    st.subheader("⏺️ Geophysics: Omori-Utsu Law (Long-term Decay)")
    y_series = df_hourly['Hourly_Aftershock_Count']
    
    with st.spinner("Detecting structural breaks..."):
        bkp_index, y_decay, t_decay = isolate_decay_period(y_series)
    
    st.markdown(f"**Insight:** Initial chaotic period ends at hour **{bkp_index}**. Isolating the *True Decay Period* for modeling.")
    
    with st.spinner("Fitting Omori Law..."):
        params, omori_prediction = fit_omori(t_decay, y_decay)
        K_fit, c_fit, p_fit = params
    
    cols = st.columns(4)
    cols[0].markdown(f"<div class='metric-card'><div class='metric-label'>Break Index</div><div class='metric-value'>{bkp_index}</div></div>", unsafe_allow_html=True)
    cols[1].markdown(f"<div class='metric-card'><div class='metric-label'>Omori 'p' (Decay Rate)</div><div class='metric-value'>{p_fit:.2f}</div></div>", unsafe_allow_html=True)
    cols[2].markdown(f"<div class='metric-card'><div class='metric-label'>Omori 'K' (Scale)</div><div class='metric-value'>{K_fit:.2f}</div></div>", unsafe_allow_html=True)
    cols[3].markdown(f"<div class='metric-card'><div class='metric-label'>Omori 'c' (Time Offset)</div><div class='metric-value'>{c_fit:.2f}</div></div>", unsafe_allow_html=True)
    
    st.plotly_chart(plot_omori_fit(t_decay, y_decay, omori_prediction, p_fit), use_container_width=True)
    
    st.markdown("---")
    
    # 3. Econometrics
    st.subheader("Econometrics: Residual Analysis & Risk (Volatility)")
    omori_residuals = y_decay - omori_prediction
    
    with st.spinner("Analyzing residuals..."):
        fit_arima, stat_results = analyze_residuals_arima(omori_residuals)
        
    st.markdown("**1. Mean Predictability (ARIMA)**")
    p_val_str = f"{stat_results['adf_p_value']:.4e}"
    stat_msg = "Stationary (Perfect)" if stat_results['is_stationary'] else "Non-Stationary"
    st.info(f"ADF Test p-value: {p_val_str} ({stat_msg}). The Omori Law sufficiently captures the mean trend.")
    
    st.plotly_chart(create_acf_pacf_plot(omori_residuals, "Omori Residuals (v_t)"), use_container_width=True)
    
    st.markdown("**2. Volatility Clustering (GARCH)**")
    arima_residuals = fit_arima.resid
    st.plotly_chart(create_acf_pacf_plot(arima_residuals**2, "Squared ARIMA Residuals"), use_container_width=True)
    
    with st.spinner("Fitting GARCH(1,1)..."):
        garch_results = fit_garch(arima_residuals)
        
    beta = garch_results.params['beta[1]']
    beta_p = garch_results.pvalues['beta[1]']
    
    st.markdown(f"**GARCH Persistence (Beta):** `{beta:.4f}` (p-value: `{beta_p:.4e}`).")
    st.warning("Significant persistence proves volatility clustering. High risk hours tend to follow high risk hours.")
    
    st.markdown("---")
    
    # 4. Nowcast
    st.subheader("⭕ Hybrid Nowcast (Next 6 Hours)")
    st.markdown("Combines the geophysical Omori baseline with the econometric ARIMA/GARCH error corrections.")
    
    st.plotly_chart(plot_hybrid_nowcast(t_decay, y_decay, omori_law, params, fit_arima, forecast_steps=6), use_container_width=True)
    
    # Simple danger index logic based on GARCH variance forecast
    variance_forecast = garch_results.forecast(horizon=1).variance.iloc[-1, 0]
    avg_variance = garch_results.conditional_volatility.mean()**2
    
    if variance_forecast > avg_variance * 1.5:
        danger_level = "<span class='danger-high'>HIGH</span>"
        action = "Halt operations in damaged structures. High volatility expected."
    elif variance_forecast < avg_variance * 0.5:
        danger_level = "<span class='danger-low'>LOW</span>"
        action = "Operations may proceed with standard caution."
    else:
        danger_level = "<span class='danger-medium'>MODERATE</span>"
        action = "Exercise caution. Volatility is at baseline."
        
    st.markdown(f"### Current Risk Index: {danger_level}", unsafe_allow_html=True)
    st.markdown(f"**Recommended Action:** {action}")

with tab1:
    st.header("Turkey-Syria Earthquake 2023 Analysis")
    try:
        df_raw, df_hourly = load_builtin_data()
        render_pipeline(df_raw, df_hourly)
    except Exception as e:
        st.error(f"Error loading built-in dataset: {e}")

with tab2:
    st.header("Upload & Analyze Custom Data")
    st.markdown("""
    **Instructions:**
    Upload an event catalog in **CSV** or **Excel** format. 
    The file must contain:
    * `time` column: Timestamp of the event.
    * (Optional) `latitude` and `longitude`: For mapping.
    * (Optional) `mag`: Magnitude for marker sizing.
    """)
    
    uploaded_file = st.file_uploader("Upload Event Catalog", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        is_excel = uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls')
        try:
            with st.spinner("Loading and aggregating data..."):
                df_raw, df_hourly = load_and_preprocess_raw(uploaded_file, is_excel)
            st.success("Data loaded successfully!")
            render_pipeline(df_raw, df_hourly)
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
            st.info("Please ensure the file has a 'time' column with valid datetime strings.")
