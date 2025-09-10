import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

def analyze_residuals_arima(residuals):
    """
    Tests stationarity and fits an ARIMA model to the residuals.
    Returns the fitted ARIMA model and the stationarity check result.
    """
    adf_result = adfuller(residuals)
    p_value = adf_result[1]
    
    # If p < 0.05, it's stationary (d=0). Otherwise non-stationary (d=1).
    d_order = 0 if p_value < 0.05 else 1
    
    model_arima = ARIMA(residuals, order=(1, d_order, 0))
    fit_arima = model_arima.fit()
    
    return fit_arima, {'adf_p_value': p_value, 'is_stationary': p_value < 0.05}

