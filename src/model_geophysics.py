import numpy as np
import ruptures as rpt
from scipy.optimize import curve_fit

def omori_law(t, K, c, p):
    return K / ((t + c)**p)

def isolate_decay_period(y_series):
    """
    Uses ruptures to find the structural break and isolate the true decay period.
    Returns:
        bkp_index (int): Index where the break occurs.
        y_decay (pd.Series): The isolated decay series.
        t_decay (np.array): The time index for the decay series (1, 2, ...).
    """
    points = y_series.values
    algo = rpt.Dynp(model="l2").fit(points)
    result = algo.predict(n_bkps=1)
    bkp_index = result[0]
    
    y_decay = y_series.iloc[bkp_index:]
    t_decay = np.arange(1, len(y_decay) + 1)
    
    return bkp_index, y_decay, t_decay

