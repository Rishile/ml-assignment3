import io
import numpy as np
import pandas as pd
import requests



# Canadian lynx trappings (1821-1934) 
def load_canadian_lynx(log_transform=True):
    url = (
        "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/"
        "master/csv/datasets/lynx.csv"
    )
    df = pd.read_csv(url, index_col=0)
    df.columns = ["year", "lynx"]
    df.index = pd.to_datetime(df["year"], format="%Y")
    series = df["lynx"].astype(float)
    if log_transform:
        series = np.log10(series)
    return series


# Sunspots (monthly)
def load_sunspots():
    from statsmodels.datasets import sunspots
    data = sunspots.load_pandas().data
    data.index = pd.date_range(start="1700", periods=len(data), freq="YE")
    return data["SUNACTIVITY"]


# Daily minimum temperatures, Melbourne 1981-1990
def load_melbourne_temperature():
    url = (
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
        "daily-min-temperatures.csv"
    )
    df = pd.read_csv(url, parse_dates=["Date"], index_col="Date")
    df.columns = ["min_temp"]
    return df


# Exchange rate (Lai et al. multivariate benchmark) 
def load_exchange_rate(country_col=0):
    """
    Columns (in order): Australia, UK, Canada, Switzerland, China, Japan,
    New Zealand, Singapore.
    """
    url = (
        "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/"
        "master/exchange_rate/exchange_rate.txt.gz"
    )
    df = pd.read_csv(url, header=None, compression="gzip")
    countries = ["Australia", "UK", "Canada", "Switzerland",
                 "China", "Japan", "New_Zealand", "Singapore"]
    df.columns = countries
    df.index = pd.date_range(start="1990-01-01", periods=len(df), freq="B")
    if country_col is None:
        return df
    return df.iloc[:, country_col]



# Household electric power consumption (UCI) 
def load_household_power(resample="h"):
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/"
        "household_power_consumption.zip"
    )
    r = requests.get(url, timeout=60)
    import zipfile
    z = zipfile.ZipFile(io.BytesIO(r.content))
    with z.open("household_power_consumption.txt") as f:
        df = pd.read_csv(f, sep=";", na_values="?", low_memory=False)
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S"
    )
    df = df.drop(columns=["Date", "Time"]).set_index("datetime")
    df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")
    series = df["Global_active_power"].dropna()
    if resample:
        series = series.resample(resample).mean()
    return series

# Beijing PM2.5 air quality (UCI) 
def load_beijing_pm25():
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/"
        "PRSA_data_2010.1.1-2014.12.31.csv"
    )
    df = pd.read_csv(url)
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.set_index("datetime")
    return df["pm2.5"].dropna()



if __name__ == "__main__":
    print(load_sunspots().head())
    print(load_canadian_lynx().head())
    print(load_melbourne_temperature().head())
    print(load_exchange_rate(country_col=1).head())
    print(load_household_power().head())
    print(load_beijing_pm25().head())