from pathlib import Path
from typing import Literal
import pandas as pd
import plotly.express as px
import yfinance as yf
import numpy as np
import polars as pl

import sys
from plotly import graph_objects as go
project_root = Path().resolve().parent
sys.path.append(str(project_root))

from scripts.rq3_function_lib import save_png

DEFAULT_DERIVED_DIR = Path(r'/Users/sebastian/data-science-projekt/tankerkoenig_data/prices')
StatType = Literal["mean", "median"]

def plot_national_fuel_prices_year(
    year: int,
    stat: StatType = "mean",
    show_oil: bool = True
):
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
    df_long["fuel_type"] = df_long["fuel_type"].str.replace(f"_{stat}_last", "", regex=False)

    fig = px.line(
        df_long,
        x="day",
        y="price",
        color="fuel_type",
        title=f"National fuel prices ({stat}) – {year}",
        labels={
            "day": "Day",
            "price": f"Price ({stat})",
            "fuel_type": "Fuel type"
        }
    )
    fig.update_traces(line=dict(width=5.5))

    if show_oil:
        oil_df = load_brent(
            start=f"{year}-01-01",
            end=f"{year+1}-01-01",
            interval="1d"
        )

        oil_df["day"] = pd.to_datetime(oil_df["time"]).dt.normalize()
        oil_df = oil_df.sort_values("day")

        fig.add_trace(
            go.Scatter(
                x=oil_df["day"],
                y=oil_df["oil_close"],
                mode="lines",
                name="Brent oil (EUR)",
                yaxis="y2",
                line=dict(
                          color="black",
                          width=7
                )
            )
        )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title=f"Price ({stat})",
        legend_title="Fuel type"
    )

    if show_oil:
        fig.update_layout(
            yaxis2=dict(
                title="Oil (EUR)",
                overlaying="y",
                side="right",
                showgrid=False
            )
        )

    save_png(fig, "oil_price_correlation.png",legend=True)
    return fig

def load_brent(start: str, end: str, interval: str = "1d",) -> pd.DataFrame: #get data from yfinance

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



if __name__ == "__main__":
    fuel_type = "diesel"
    year = 2025
    month = 12

    df = pl.read_parquet(
        f"/Users/sebastian/data-science-projekt/tankerkoenig_data/prices/price_change_events_{year}_{month:02d}.parquet"
    )
    df = df.filter(pl.col("fuel_type") == fuel_type)
    df = df.sort(["station_id", "fuel_type", "timestamp"])
    df = df.with_columns(
        pl.col("price_change")
        .cum_sum()
        .over(["station_id", "fuel_type"])
        .alias("price_index")
    )

    df = df.with_columns(
        (
            pl.col("price_index")
            - pl.col("price_index").mean().over(["station_id", "fuel_type"])
        ).alias("price_norm")
    )
    hour_weekday_price = (
        df.group_by(["weekday", "hour"])
        .agg(pl.mean("price_norm").alias("mean_price"))
        .sort(["weekday", "hour"])
    )
    hour_weekday_price = hour_weekday_price.with_columns(
        pl.when(pl.col("weekday") == 1).then(pl.lit("Mon"))
        .when(pl.col("weekday") == 2).then(pl.lit("Tue"))
        .when(pl.col("weekday") == 3).then(pl.lit("Wed"))
        .when(pl.col("weekday") == 4).then(pl.lit("Thu"))
        .when(pl.col("weekday") == 5).then(pl.lit("Fri"))
        .when(pl.col("weekday") == 6).then(pl.lit("Sat"))
        .when(pl.col("weekday") == 7).then(pl.lit("Sun"))
        .otherwise(pl.lit("Unknown"))
        .alias("weekday_label")
    )
    plot_df = hour_weekday_price.to_pandas()

    order = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

    pivot_df = plot_df.pivot(index="weekday_label", columns="hour", values="mean_price")
    pivot_df = pivot_df.reindex(order)

    fig = px.imshow(
        pivot_df,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        origin="lower",
        title=f"Intraday Fuel Price Cycle (Normalized) - {fuel_type.upper()}"
    )

    fig.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Weekday",
        margin=dict(r=140),
        coloraxis_colorbar=dict(
        title=dict(
            text="Normalized Price",
            font=dict(size=75)
        ),
        tickfont=dict(size=70),
        thickness=45,
        len=0.95,
        x=1.02
    )
    )

    save_png(fig, img_name="rq1_best_time_heatmap.png" )
    fig.show()