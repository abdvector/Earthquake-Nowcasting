import pandas as pd
import numpy as np

def load_and_preprocess_raw(file_obj, is_excel=False):
    """
    Loads raw earthquake catalog (CSV or Excel), converts timestamps,
    and aggregates into hourly counts.
    Expected columns: 'time', 'latitude', 'longitude', 'mag'
    """
    if is_excel:
        df = pd.read_excel(file_obj)
    else:
        df = pd.read_csv(file_obj)
        
    # Ensure time is datetime
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df.sort_values('time', inplace=True)
    else:
        raise ValueError("The uploaded dataset must contain a 'time' column.")
        
    # Set time as index for aggregation
    df.set_index('time', inplace=True)
    
    # Aggregate to hourly
    # We count the number of events per hour
    hourly_counts = df.resample('h').size().rename('Hourly_Aftershock_Count').to_frame()
    
    # We can also keep average lat/lon per hour if present for mapping, or just keep raw for mapping
    # But for the map, raw data is better. So we return both.
    
    # Drop rows with NaNs in hourly counts (shouldn't happen with size(), but just in case)
    hourly_counts.fillna(0, inplace=True)
    
    return df.reset_index(), hourly_counts

def load_builtin_data():
    raw_file = "Turkey_Aftershops_2023.csv.csv"
    return load_and_preprocess_raw(raw_file, is_excel=False)
