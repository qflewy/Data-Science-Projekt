from pathlib import Path
import polars as pl
import pandas as pd
import plotly.graph_objects as go

def load_stations(stations_csv: Path, stations_parquet: Path) -> pl.DataFrame:
    """
    Load stations data and cache it as parquet for faster reload.
    """

    if stations_parquet.exists():
        return pl.read_parquet(stations_parquet)

    stations = (
        pl.scan_csv(stations_csv)
        .select(["uuid", "brand", "city", "latitude", "longitude"])
        .collect()
    )

    stations.write_parquet(stations_parquet)
    return stations

def anomalies_per_station(combined: pl.DataFrame) -> pl.DataFrame:
    """
    Count anomaly events per station.
    """

    return (
        combined
        .group_by("station_uuid")
        .agg(pl.len().alias("anomalies"))
    )

def total_updates(prices_path: Path, start_year: int, end_year: int) -> pl.DataFrame:
    """
    Count total updates and anomalies (diesel >= e10) per station.
    Only rows where both diesel > 0 and e10 > 0 are considered (excludes diesel-only stations).
    """
    import gc
    from collections import defaultdict

    all_files = []
    for year in range(start_year, end_year):
        year_path = prices_path / str(year)
        if year_path.exists():
            all_files.extend(sorted(year_path.glob("*.parquet")))

    if not all_files:
        return pl.DataFrame({"station_uuid": [], "updates": [], "anomalies": []})

    update_counts: dict = defaultdict(int)
    anomaly_counts: dict = defaultdict(int)

    for i, f in enumerate(all_files):
        df = pd.read_parquet(str(f), columns=["station_uuid", "diesel", "e10"])
        valid = df[(df["diesel"].notna()) & (df["e10"].notna()) & (df["diesel"] > 0) & (df["e10"] > 0)]
        for uuid, cnt in valid.groupby("station_uuid").size().items():
            update_counts[uuid] += int(cnt)
        for uuid, cnt in valid[valid["diesel"] >= valid["e10"]].groupby("station_uuid").size().items():
            anomaly_counts[uuid] += int(cnt)
        del df, valid
        gc.collect()
        if (i + 1) % 12 == 0:
            print(f"  {i + 1}/{len(all_files)} files done")

    return pl.DataFrame({
        "station_uuid": list(update_counts.keys()),
        "updates": [update_counts[k] for k in update_counts],
        "anomalies": [anomaly_counts.get(k, 0) for k in update_counts],
    })

def save_png(fig, img_name:Path, legend:bool=False):

    px_w = 3300
    px_h = 1650

    fig = fig.full_figure_for_development(warn = False)
    fig.update_layout(
        autosize = False,
        width = px_w,
        height = px_h,
        font = dict(size=44),
        title = dict(font = dict(size=50),
                     y = .98,
                     x = .5,
                     xanchor = "center",
                     yanchor = "top")
    )
    if legend:
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                font = dict(size = 38),
                y= .92,
                xanchor="left",
                x=0
            )
        )

    fig.update_xaxes(tickfont = dict(size = 38), title_font = dict(size = 40))
    fig.update_yaxes(tickfont = dict(size = 38), title_font = dict(size = 40))

    img_path = Path(r'D:/Tankdaten/rq_brand_vs_free')
    fig.write_image(img_path/img_name,
                    width=px_w,
                    height=px_h,
                    scale=1)
    
def anomaly_rate(updates: pl.DataFrame) -> pl.DataFrame:
    """
    Compute anomaly rate per station.
    """

    return updates.with_columns(
        (pl.col("anomalies") / pl.col("updates")).alias("anomaly_rate")
    )


def plot_anomaly_rate(combined: pl.DataFrame, parquet_base: Path, start_year: int, end_year: int) -> go.Figure:
    """
    Compute monthly anomaly rate (anomalies / total price updates)
    """
    anomalies_per_month = (
        combined.group_by(["year", "month"]).agg(pl.len().alias("count")).sort(["year", "month"])
    )

    monthly_updates_list = []
    for year in range(start_year, end_year):
        year_dir = parquet_base / str(year)
        if not year_dir.exists():
            continue
        counts = (
            pl.scan_parquet(str(year_dir / "*.parquet"))
            .select(pl.col("date").str.slice(0, 7).alias("ym"))
            .group_by("ym")
            .agg(pl.len().alias("total_updates"))
            .collect()
        )
        monthly_updates_list.append(counts)
        print(f"{year}: {counts['total_updates'].sum():,} updates")

    monthly_updates = (
        pl.concat(monthly_updates_list)
        .with_columns([
            pl.col("ym").str.slice(0, 4).cast(pl.Int32).alias("year"),
            pl.col("ym").str.slice(5, 2).cast(pl.Int32).alias("month"),
        ])
        .drop("ym")
    )

    joined = (
        anomalies_per_month
        .rename({"count": "anomalies"})
        .join(monthly_updates.rename({"total_updates": "updates"}), on=["year", "month"], how="inner")
        .sort(["year", "month"])
    )
    rate_df = anomaly_rate(joined).to_pandas()
    rate_df["date"] = pd.to_datetime(rate_df[["year", "month"]].assign(day=1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rate_df["date"],
        y=rate_df["anomaly_rate"] * 100, 
        mode="lines+markers",
        marker=dict(size=5),
        line=dict(width=2, color="steelblue"),
        hovertemplate="%{x|%Y-%m}<br>Rate: %{y:.2f}%<extra></extra>" # with help from Claude Code
    ))
    fig.update_layout(
        title=f"E10/Diesel Anomaly Rate per Month ({start_year}-{end_year - 1})",
        xaxis_title="Date",
        yaxis_title="Anomaly Rate (%)",
        hovermode="x unified",
        xaxis=dict(tickformat="%Y-%m", tickangle=45),
        template="plotly_white"
    )
    return fig

def plot_anomalies_per_year(combined: pl.DataFrame, start_year: int, end_year: int) -> go.Figure:
    """
    Plot total anomalies per year as an interactive Plotly bar chart.
    """
    year_data = (
        combined.group_by("year").agg(pl.len().alias("count")).sort("year").to_pandas()
    )
    year_data["count_m"] = year_data["count"] / 1_000_000

    fig = go.Figure(go.Bar(
        x=year_data["year"],
        y=year_data["count_m"],
        marker_color="steelblue",
        opacity=0.7,
        text=year_data["count_m"].map("{:.1f}M".format),
        textposition="outside",
        hovertemplate="Year: %{x}<br>Anomalies: %{text}<extra></extra>", # Claude Code
    ))
    fig.update_layout(
        title=f"Total Anomalies per Year ({start_year}-{end_year - 1})",
        xaxis_title="Year",
        yaxis_title="Anomalies (Millions)",
        xaxis=dict(tickmode="linear"),
        bargap=0.3,
    )
    return fig


def plot_anomaly_rate_by_hour(parquet_base: Path, year: int) -> go.Figure:
    """
    Plot anomaly rate (diesel >= e10) by hour of day for a given year as an interactive Plotly bar chart.
    """
    lf = pl.scan_parquet(str(parquet_base / str(year) / "*.parquet"))

    # Extract hour via string slice (positions 11-12 = "HH") — avoids regex+datetime parsing crash on large files
    lf = lf.with_columns([
        pl.col("date").str.slice(11, 2).cast(pl.Int32).alias("hour"),
        (
            pl.col("diesel").is_not_null() &
            pl.col("e10").is_not_null() &
            (pl.col("diesel") >= pl.col("e10"))
        ).alias("anomaly")
    ])

    hourly = (
        lf.group_by("hour")
        .agg([
            pl.len().alias("updates"),
            pl.col("anomaly").sum().alias("anomalies")
        ])
        .with_columns(
            (pl.col("anomalies") / pl.col("updates")).alias("anomaly_rate")
        )
        .sort("hour")
        .collect()
    )

    fig = go.Figure(go.Bar(
        x=hourly["hour"].to_list(),
        y=hourly["anomaly_rate"].to_list(),
        marker_color="steelblue",
        hovertemplate="Hour: %{x}<br>Anomaly Rate: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Anomaly Rate by Hour of Day ({year})",
        xaxis_title="Hour of Day",
        yaxis_title="Anomaly Rate",
        xaxis=dict(tickmode="linear"),
    )
    return fig


def prepare_station_map(station_stats: pl.DataFrame, stations: pl.DataFrame, top_n: int = 15) -> pl.DataFrame:
    """
    Prepare dataframe for plotting map.
    """

    top = (
        station_stats
        .sort("anomaly_rate", descending=True)
        .head(top_n)
    )

    joined = top.join(
        stations,
        left_on="station_uuid",
        right_on="uuid",
        how="left"
    )

    return joined.with_columns(
        (pl.col("brand").fill_null("Unknown") + " " + pl.col("city").fill_null(""))
        .alias("name")
    )


def plot_top_stations_map(stations_csv: Path, stations_parquet: Path, stats_cache: Path, start_year: int, end_year: int, top_n: int = 100, min_updates: int = 50000) -> go.Figure:
    """
    Load station data and stats, then plot the top N stations by anomaly rate.
    """
    stations = load_stations(stations_csv, stations_parquet)
    station_stats = pl.read_parquet(stats_cache).filter(pl.col("updates") >= min_updates)
    map_df = prepare_station_map(station_stats, stations, top_n=top_n)
    station_data = map_df.to_pandas()

    import plotly.express as px
    fig = px.scatter_mapbox(
        station_data,
        lat="latitude",
        lon="longitude",
        size="anomaly_rate",
        color="anomaly_rate",
        hover_name="name",
        hover_data={"anomalies": True, "updates": True, "anomaly_rate": ":.3f"},
        size_max=18,
        zoom=5,
        center=dict(lat=51.5, lon=10.0),
        height=600,
        title=f"Top {top_n} Gas Stations by Anomaly Rate ({start_year}-{end_year - 1})",
        color_continuous_scale="Turbo",
        mapbox_style="open-street-map",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    return fig, map_df