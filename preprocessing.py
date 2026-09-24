import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Missing values
def missing_value_report(series: pd.Series) -> dict:
    n_missing = series.isna().sum()
    pct_missing = 100 * n_missing / len(series)
    return {
        "n_total": len(series),
        "n_missing": int(n_missing),
        "pct_missing": round(float(pct_missing), 3),
    }

def handle_missing(series: pd.Series, method: str = "interpolate") -> pd.Series:
    if method == "interpolate":
        return series.interpolate(method="time" if isinstance(series.index, pd.DatetimeIndex)
                                   else "linear")
    elif method == "ffill":
        return series.ffill()
    elif method == "drop":
        return series.dropna()
    else:
        raise ValueError(f"Unknown method: {method}")

# Stationarity (Augmented Dickey-Fuller test)
def stationarity_report(series: pd.Series, alpha: float = 0.05) -> dict:
    clean = series.dropna().values
    adf_stat, p_value, used_lag, n_obs, crit_values, _ = adfuller(clean)
    return {
        "adf_stat": round(float(adf_stat), 4),
        "p_value": round(float(p_value), 4),
        "used_lag": int(used_lag),
        "n_obs": int(n_obs),
        "critical_values": {k: round(v, 4) for k, v in crit_values.items()},
        "is_stationary": bool(p_value < alpha),
    }

def difference_series(series: pd.Series, order: int = 1) -> pd.Series:
    """First-difference (or higher order) the series to remove trend."""
    return series.diff(order).dropna()

# Chronological train/val/test split
def chronological_split(series: pd.Series, train_frac=0.7, val_frac=0.15):
    n = len(series)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train = series.iloc[:train_end]
    val = series.iloc[train_end:val_end]
    test = series.iloc[val_end:]
    return train, val, test

# Scaling
def scale_series(train, val, test, method="minmax"):
    scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
    train_arr = train.values.reshape(-1, 1)
    val_arr = val.values.reshape(-1, 1)
    test_arr = test.values.reshape(-1, 1)

    scaled_train = scaler.fit_transform(train_arr).flatten()
    scaled_val = scaler.transform(val_arr).flatten()
    scaled_test = scaler.transform(test_arr).flatten()
    return scaled_train, scaled_val, scaled_test, scaler

# Windowing
def make_windows(series_values: np.ndarray, window_size: int, horizon: int = 1):
    X, y = [], []
    for i in range(len(series_values) - window_size - horizon + 1):
        X.append(series_values[i: i + window_size])
        y.append(series_values[i + window_size: i + window_size + horizon])
    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y)
    return X, y

def make_windows_from_series(series: pd.Series, window_size: int, horizon: int = 1,
                              freq: str = None):
    idx = series.index
    values = series.to_numpy()
 
    if freq is None:
        step = idx.to_series().diff().dropna().mode().iloc[0]
    else:
        step = pd.tseries.frequencies.to_offset(freq)
 
    diffs = idx.to_series().diff()
    is_break = (diffs != step)
    is_break.iloc[0] = True  # first point always starts a new segment
    segment_id = is_break.cumsum().to_numpy()
 
    X_list, y_list = [], []
    for seg in np.unique(segment_id):
        seg_values = values[segment_id == seg]
        if len(seg_values) < window_size + horizon:
            continue  # segment too short to form even one window -- skipped
        Xs, ys = make_windows(seg_values, window_size, horizon)
        X_list.append(Xs)
        y_list.append(ys)
 
    if not X_list:
        return np.empty((0, window_size, 1)), np.empty((0, horizon))
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

if __name__ == "__main__":
    from load_datasets import (
    load_beijing_pm25,
    load_canadian_lynx,
    load_exchange_rate,
    load_household_power,
    load_melbourne_temperature,
    load_sunspots,
    )

    sunspots = load_sunspots()
    lynx = load_canadian_lynx()
    temp = load_melbourne_temperature()
    exchange = load_exchange_rate(country_col=1)
    power = load_household_power()
    beijing = load_beijing_pm25()

    # Missing values
    power_clean = handle_missing(power, method="drop")

    # Stationarity
    sunspots_diff = difference_series(sunspots)
    exchange_diff = difference_series(exchange)

    # Train/Test split
    sunspots_train, sunspots_val, sunspots_test = chronological_split(sunspots_diff)
    lynx_train, lynx_val, lynx_test = chronological_split(lynx)
    temp_train, temp_val, temp_test = chronological_split(temp)
    exchange_train, exchange_val, exchange_test = chronological_split(exchange_diff)
    power_train, power_val, power_test = chronological_split(power_clean)

    beijing_log = np.log1p(beijing)
    beijing_train, beijing_val, beijing_test = chronological_split(beijing)

    datasets = {
    "sunspots":  (sunspots_train, sunspots_val, sunspots_test, "minmax"),
    "lynx":      (lynx_train, lynx_val, lynx_test, "minmax"),
    "temp":      (temp_train, temp_val, temp_test, "minmax"),
    "exchange":  (exchange_train, exchange_val, exchange_test, "minmax"),
    "power":     (power_train, power_val, power_test, "standard"),
    "beijing":   (beijing_train, beijing_val, beijing_test, "standard"),
    }

    scaled = {}

    for name, (tr, va, te, method) in datasets.items():
        tr_s, va_s, te_s, scaler = scale_series(tr, va, te, method=method)
        scaled[name] = {
            "train": tr_s, "val": va_s, "test": te_s, "scaler": scaler,
        }

    window_sizes = {
    "sunspots": 12,
    "lynx":     10,
    "temp":     14,
    "exchange": 20,
    "power":    24,
    "beijing":  24,
    }

    windowed = {}

    for name, ws in window_sizes.items():
        tr_arr, va_arr, te_arr, scaler = scaled[name] 

        if name == "power":
            tr_idx = power_train.index
            va_idx = power_val.index
            te_idx = power_test.index
            tr_s = pd.Series(tr_arr, index=tr_idx)
            va_s = pd.Series(va_arr, index=va_idx)
            te_s = pd.Series(te_arr, index=te_idx)
            X_train, y_train = make_windows_from_series(tr_s, ws, horizon=1, freq="h")
            X_val,   y_val   = make_windows_from_series(va_s, ws, horizon=1, freq="h")
            X_test,  y_test  = make_windows_from_series(te_s, ws, horizon=1, freq="h")
        else:
            # no gaps; plain windowing on the array
            X_train, y_train = make_windows(tr_arr, ws, horizon=1)
            X_val,   y_val   = make_windows(va_arr, ws, horizon=1)
            X_test,  y_test  = make_windows(te_arr, ws, horizon=1)

        windowed[name] = {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
        }
        print(f"{name}: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
