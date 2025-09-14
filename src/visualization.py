import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf

def plot_omori_fit(t_decay, y_decay, omori_prediction, p_fit):
    """
    Plots the actual vs predicted Omori fit.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=t_decay, y=y_decay,
        mode='lines',
        name='Actual Counts (Decay Period)',
        line=dict(color='rgba(135, 206, 250, 0.7)', width=1.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=t_decay, y=omori_prediction,
        mode='lines',
        name=f'Omori Law Fit (p={p_fit:.2f})',
        line=dict(color='#ff4b4b', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Omori Law Fit to the "True" Decay Period',
        xaxis_title='Hours since start of decay',
        yaxis_title='Aftershock Count',
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def create_acf_pacf_plot(series, title_prefix):
    """
    Creates an ACF/PACF side-by-side plot using plotly.
    """
    # Calculate ACF and PACF
    nlags = min(40, len(series)//2 - 1)
    acf_vals, acf_confint = acf(series, nlags=nlags, alpha=0.05)
    pacf_vals, pacf_confint = pacf(series, nlags=nlags, alpha=0.05, method='ywm')
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f'ACF of {title_prefix}', f'PACF of {title_prefix}'))
    
    # ACF
    fig.add_trace(go.Bar(x=np.arange(len(acf_vals)), y=acf_vals, name='ACF', marker_color='#4169E1'), row=1, col=1)
    # Add confidence intervals
    lower_acf = acf_confint[:, 0] - acf_vals
    upper_acf = acf_confint[:, 1] - acf_vals
    
    fig.add_trace(go.Scatter(x=np.arange(len(acf_vals)), y=lower_acf, mode='lines', line=dict(color='rgba(255,255,255,0.2)'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(len(acf_vals)), y=upper_acf, mode='lines', fill='tonexty', fillcolor='rgba(255,255,255,0.1)', line=dict(color='rgba(255,255,255,0.2)'), showlegend=False), row=1, col=1)

    # PACF
    fig.add_trace(go.Bar(x=np.arange(len(pacf_vals)), y=pacf_vals, name='PACF', marker_color='#20B2AA'), row=1, col=2)
    # Add confidence intervals
    lower_pacf = pacf_confint[:, 0] - pacf_vals
    upper_pacf = pacf_confint[:, 1] - pacf_vals
    
    fig.add_trace(go.Scatter(x=np.arange(len(pacf_vals)), y=lower_pacf, mode='lines', line=dict(color='rgba(255,255,255,0.2)'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=np.arange(len(pacf_vals)), y=upper_pacf, mode='lines', fill='tonexty', fillcolor='rgba(255,255,255,0.1)', line=dict(color='rgba(255,255,255,0.2)'), showlegend=False), row=1, col=2)
    
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def plot_hybrid_nowcast(t_decay, y_decay, omori_law_fn, params, fit_arima, forecast_steps=6):
    """
    Plots the final hybrid nowcast.
    """
    K_fit, c_fit, p_fit = params
    last_t = len(y_decay)
    
    # Omori forecast
    forecast_index = np.arange(last_t + 1, last_t + 1 + forecast_steps)
    omori_forecast = omori_law_fn(forecast_index, K_fit, c_fit, p_fit)
    
    # ARIMA forecast
    arima_forecast = fit_arima.get_forecast(steps=forecast_steps).predicted_mean
    
    # Hybrid
    hybrid_nowcast = omori_forecast + arima_forecast
    
    fig = go.Figure()
    
    # Plot last 48 hours of actual data
    lookback = min(48, len(y_decay))
    t_lookback = t_decay[-lookback:]
    y_lookback = y_decay.values[-lookback:]
    
    fig.add_trace(go.Scatter(
        x=t_lookback, y=y_lookback,
        mode='lines',
        name='Actual Counts (Last 48h)',
        line=dict(color='rgba(135, 206, 250, 0.9)', width=2)
    ))
    
    # Connect forecast to last actual point
    plot_forecast_index = np.insert(forecast_index, 0, last_t)
    plot_hybrid_vals = np.insert(hybrid_nowcast.values, 0, y_lookback[-1])
    
    fig.add_trace(go.Scatter(
        x=plot_forecast_index, y=plot_hybrid_vals,
        mode='lines+markers',
        name=f'Hybrid Nowcast (Next {forecast_steps} Hours)',
        line=dict(color='#ff4b4b', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title='Hybrid Econometric-Geophysical Nowcast',
        xaxis_title='Hours since start of decay',
        yaxis_title='Aftershock Count',
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.5)'),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig
