from pathlib import Path
from typing import Literal, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import numpy as np

# path for folder:
DEFAULT_DERIVED_DIR = Path(r"D:\data\derived\national_daily_last") # +++Path+++

StatType = Literal["mean", "median"]

def plot_national_fuel_prices_year(year: int, stat: str = "mean"):

    if stat not in ("mean", "median"):
        raise ValueError("stat must be 'mean' or 'median'")

    file_path = DEFAULT_DERIVED_DIR / f"national_daily_last_{year}.csv"
    df = pd.read_csv(file_path)

    df["day"] = pd.to_datetime(df["day"])
    df = df.sort_values("day")
    fuels = ["e5", "e10", "diesel"]
    value_cols = [f"{fuel}_{stat}_last" for fuel in fuels]
    df_long = df.melt(
        id_vars="day",
        value_vars=value_cols,
        var_name="fuel_type",
        value_name="price"
    )
    df_long["fuel_type"] = df_long["fuel_type"].str.replace(f"_{stat}_last", "")

    plt.figure(figsize=(12,6))
    sns.lineplot(data=df_long, x="day", y="price", hue="fuel_type")
    plt.title(f"National fuel prices ({stat}) – {year}")
    plt.xlabel("day")
    plt.ylabel(f"price ({stat})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()



def load_brent(start: str, end: str, interval: str = "1d",) -> pd.DataFrame:

    if interval not in ("1d", "1h"):
        raise ValueError("interval must be '1d' or '1h'")

    ticker = "BZ=F"  # Brent futures
    df = yf.Ticker(ticker).history(start=start, end=end, interval=interval)

    if df is None or df.empty:
        raise RuntimeError(
            f"No data returned for Brent ({ticker}) with start={start}, end={end}, interval={interval}. "
            "Yahoo may limit the lookback for intraday intervals (e.g., 1h)."
        )
    # Reset index to column; name may be 'Date' or 'Datetime' depending on interval
    df = df.reset_index()
    time_col = "Datetime" if "Datetime" in df.columns else "Date"
    out = df[[time_col, "Close"]].rename(columns={time_col: "time", "Close": "oil_close"})

    # Make timezone-naive
    out["time"] = pd.to_datetime(out["time"]).dt.tz_localize(None)
    # Sort
    out = out.sort_values("time").reset_index(drop=True)
    # ---------------------------------------------------
    # Load EUR/USD exchange rate
    # ---------------------------------------------------
    fx = yf.Ticker("EURUSD=X").history(start=start, end=end, interval=interval)
    if fx is None or fx.empty:
        raise RuntimeError("No FX data returned for EURUSD.")
    fx = fx.reset_index()
    fx_time_col = "Datetime" if "Datetime" in fx.columns else "Date"
    fx = fx[[fx_time_col, "Close"]].rename(
        columns={fx_time_col: "time", "Close": "eurusd"}
    )
    fx["time"] = pd.to_datetime(fx["time"]).dt.tz_localize(None)
    fx = fx.sort_values("time")
    # ---------------------------------------------------
    # Merge Brent and FX (robust for hourly data)
    # ---------------------------------------------------
    merged = pd.merge_asof(
        out.sort_values("time"),
        fx.sort_values("time"),
        on="time",
        direction="backward"
    )
    merged["eurusd"] = merged["eurusd"].ffill().bfill()
    # ---------------------------------------------------
    # Convert USD → EUR
    # ---------------------------------------------------
    merged["oil_close"] = merged["oil_close"] / merged["eurusd"]
    out = merged[["time", "oil_close"]]

    return out



def set_plot_style():

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="darkgrid")

    plt.rcParams.update({
        "figure.facecolor": "#1e1e1e",
        "axes.facecolor": "#252526",
        "axes.edgecolor": "#cccccc",
        "axes.labelcolor": "#cccccc",
        "text.color": "#cccccc",
        "xtick.color": "#cccccc",
        "ytick.color": "#cccccc",
        "grid.color": "#444444",
        "grid.alpha": 0.4,
        "axes.titlecolor": "#ffffff",
    })



def load_merge_year(year: int, fuel_type: str, interval: str = "1d") -> pd.DataFrame:
    """
    Load one year of national fuel data + Brent oil, merge on day, forward-fill oil.
    Returns columns: day, fuel_price, oil_close
    """
    oil = load_brent(f"{year}-01-01", f"{year+1}-01-01", interval=interval)
    fuel_path = DEFAULT_DERIVED_DIR / f"national_daily_last_{year}.csv"
    fuel = pd.read_csv(fuel_path)

    oil["day"] = pd.to_datetime(oil["time"]).dt.normalize()
    oil = oil[["day", "oil_close"]]

    fuel["day"] = pd.to_datetime(fuel["day"]).dt.date
    oil["day"] = pd.to_datetime(oil["day"]).dt.tz_localize(None).dt.date

    fuel = fuel[["day", fuel_type]].rename(columns={fuel_type: "fuel_price"})

    merged = fuel.merge(oil, on="day", how="left").sort_values("day")
    merged["oil_close"] = merged["oil_close"].ffill()
    merged = merged.dropna(subset=["fuel_price", "oil_close"]).reset_index(drop=True)

    return merged


def add_returns_and_lags(df: pd.DataFrame, K_LAG: int) -> pd.DataFrame:
    """
    Add log-returns r_fuel, r_oil and lag columns r_oil_lag0..K_LAG.
    Drops NaNs created by diff/shift.
    Expects columns: fuel_price, oil_close
    """
    out = df.copy()

    out["r_fuel"] = np.log(out["fuel_price"]).diff()
    out["r_oil"]  = np.log(out["oil_close"]).diff()

    out = out.dropna(subset=["r_fuel", "r_oil"]).reset_index(drop=True)

    lags = pd.concat([out["r_oil"].shift(k) for k in range(K_LAG + 1)], axis=1)
    lags.columns = [f"r_oil_lag{k}" for k in range(K_LAG + 1)]

    out = pd.concat([out, lags], axis=1)

    lag_cols = list(lags.columns)
    out = out.dropna(subset=lag_cols + ["r_fuel"]).reset_index(drop=True)

    return out


def compute_ccf_from_prepared(prep: pd.DataFrame, K_LAG: int) -> pd.Series:
    """
    Compute CCF = corr(r_fuel_t, r_oil_{t-k}) for k=0..K_LAG from prepared dataframe.
    Expects columns r_fuel and r_oil_lag0..K_LAG.
    """
    lag_cols = [f"r_oil_lag{k}" for k in range(K_LAG + 1)]
    return prep[lag_cols].corrwith(prep["r_fuel"])

def compute_ccf_year(year: int, fuel_type: str = "e5_mean_last", K_LAG: int = 50) -> pd.Series:
    merged = load_merge_year(year, fuel_type=fuel_type, interval="1d")
    prep = add_returns_and_lags(merged, K_LAG=K_LAG)
    corr = compute_ccf_from_prepared(prep, K_LAG=K_LAG)
    corr.index = [f"lag{k}" for k in range(K_LAG + 1)]  # optional
    return corr