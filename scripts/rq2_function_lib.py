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

    # Make timezone-naive for merging with your fuel 'day'
    out["time"] = pd.to_datetime(out["time"]).dt.tz_localize(None)

    # Sort
    out = out.sort_values("time").reset_index(drop=True)

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


def compute_ccf_year(year, fuel_type="e5_mean_last", K_LAG=50):

    oil = load_brent(f"{year}-01-01", f"{year+1}-01-01", interval="1d")
    fuel_path = DEFAULT_DERIVED_DIR / f"national_daily_last_{year}.csv"
    fuel = pd.read_csv(fuel_path)

    oil["day"] = pd.to_datetime(oil["time"]).dt.normalize()
    oil = oil[["day","oil_close"]]
    fuel["day"] = pd.to_datetime(fuel["day"]).dt.date
    oil["day"] = pd.to_datetime(oil["day"]).dt.tz_localize(None).dt.date
    fuel = fuel[["day", fuel_type]]

    merged = fuel.merge(oil,on="day",how="left").sort_values("day")
    merged["oil_close"] = merged["oil_close"].ffill()
    merged = merged.dropna()
    df = merged.copy()

    df["r_fuel"] = np.log(df[fuel_type]).diff()
    df["r_oil"] = np.log(df["oil_close"]).diff()
    df = df.dropna()

    lags = pd.concat([df["r_oil"].shift(k) for k in range(K_LAG+1)], axis=1)
    lags.columns = [f"lag{k}" for k in range(K_LAG+1)]
    df = pd.concat([df,lags],axis=1)
    data = df.dropna()
    corr = data[lags.columns].corrwith(data["r_fuel"])

    return corr


def load_and_merge_year(
    year: int,
    fuel_type: str,
    fuel_dir: Path = DEFAULT_DERIVED_DIR,
    interval: str = "1d",
) -> pd.DataFrame:
    # --- load oil ---
    oil = load_brent(f"{year}-01-01", f"{year+1}-01-01", interval=interval)
    oil["day"] = pd.to_datetime(oil["time"]).dt.normalize()
    oil = oil[["day", "oil_close"]]
    oil["day"] = pd.to_datetime(oil["day"]).dt.tz_localize(None).dt.date

    # --- load fuel ---
    fuel_dir = Path(fuel_dir)
    fuel_path = fuel_dir / f"national_daily_last_{year}.csv"
    fuel = pd.read_csv(fuel_path)
    fuel["day"] = pd.to_datetime(fuel["day"]).dt.date
    fuel = fuel[["day", fuel_type]].rename(columns={fuel_type: "fuel_price"})

    # --- merge + fill ---
    merged = fuel.merge(oil, on="day", how="left").sort_values("day")
    merged["oil_close"] = merged["oil_close"].ffill()
    merged = merged.dropna(subset=["fuel_price", "oil_close"]).reset_index(drop=True)

    return merged


def build_panel_dataset(
    years,
    fuel_type: str,
    fuel_dir: Path = DEFAULT_DERIVED_DIR,
) -> pd.DataFrame:
    out = []
    for y in years:
        df_y = load_and_merge_year(y, fuel_type=fuel_type, fuel_dir=fuel_dir)
        df_y["year"] = y
        out.append(df_y)

    return pd.concat(out, ignore_index=True)