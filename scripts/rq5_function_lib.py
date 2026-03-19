from pathlib import Path
import polars as pl
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import gc
from collections import defaultdict


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
    Uses pandas with file-by-file iteration instead of Polars scan_parquet to avoid
    loading all years into memory at once (Multiple Crashes then asked Claude Code how to fix that and he suggested this and it worked)
    """

    all_files = []
    for year in range(start_year, end_year):
        year_path = prices_path / str(year)
        if year_path.exists():
            all_files.extend(sorted(year_path.glob("*.parquet")))

    if not all_files:
        return pl.DataFrame({"station_uuid": [], "updates": [], "anomalies": []})

    update_counts: dict = defaultdict(int)
    anomaly_counts: dict = defaultdict(int)

    for f in all_files:
        df = pd.read_parquet(str(f), columns=["station_uuid", "diesel", "e10"])

        valid = df[(df["diesel"].notna()) & (df["e10"].notna()) & (df["diesel"] > 0) & (df["e10"] > 0)]

        for uuid, cnt in valid.groupby("station_uuid").size().items():
            update_counts[uuid] += int(cnt)

        for uuid, cnt in valid[valid["diesel"] >= valid["e10"]].groupby("station_uuid").size().items():
            anomaly_counts[uuid] += int(cnt)

        del df, valid
        gc.collect()

    return pl.DataFrame({
        "station_uuid": list(update_counts.keys()),
        "updates": [update_counts[k] for k in update_counts],
        "anomalies": [anomaly_counts.get(k, 0) for k in update_counts],
    })

def save_png(fig, img_name:Path, legend:bool=False):
    """
    This method saves plotly figures with high resolution (for the poster) to the given output path. Before saving the plot,
    the method adjusts the text to an appropriate size. If the plot has legend, set legend to True so it also adjusts the legends font size before saving. The chosen figure name should contain the suffix ".png". Returns nothing.
    """

    px_w = 3700
    px_h = 2250

    fig = fig.full_figure_for_development(warn = False)
    fig.update_layout(
        autosize = False,
        width = px_w/2,
        height = px_h,
        font = dict(size=94),
        title = dict(font = dict(size =100),
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
                font = dict(size = 85),
                y= .92,
                xanchor="left",
                itemsizing="constant",
                x=0
            ),
            #legend_title = None
        )

    fig.update_xaxes(tickfont = dict(size = 80), title_font = dict(size = 90))
    fig.update_yaxes(tickfont = dict(size = 80), title_font = dict(size = 90))

    img_path = Path(r'D:\Bilder') #change path to own pc
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
    Compute monthly anomaly rate (anomalies / total price updates).
    Result is cached as a compressed parquet file next to the anomaly files for WebApp.
    """
    cache_path = parquet_base / f"monthly_anomaly_rate_{start_year}_{end_year - 1}.parquet"

    if cache_path.exists():
        rate_df = pl.read_parquet(cache_path).to_pandas()
    else:
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
        rate_pl = anomaly_rate(joined).select(["year", "month", "anomalies", "updates", "anomaly_rate"])
        rate_pl.write_parquet(cache_path, compression="zstd", compression_level=19)
        print(f"Cache saved: {cache_path.name}")
        rate_df = rate_pl.to_pandas()

    rate_df["date"] = pd.to_datetime(rate_df[["year", "month"]].assign(day=1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rate_df["date"],
        y=rate_df["anomaly_rate"] * 100, 
        mode="lines+markers",
        marker=dict(size=5),
        line=dict(width=2, color="steelblue"),
        hovertemplate="%{x|%Y-%m}<br>Rate: %{y:.2f}%<extra></extra>" # from Claude Code
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


def precompute_anomaly_rate_by_hour(parquet_base: Path, anomalies_path: Path, years: list[int]) -> None:
    """
    Precompute hourly anomaly rate for each year and save as small parquet files in anomalies_path for WebApp.
    """
    for year in years:
        cache_path = anomalies_path / f"hourly_anomaly_rate_{year}.parquet"
        if cache_path.exists():
            continue
        lf = pl.scan_parquet(str(parquet_base / str(year) / "*.parquet"))
        hourly = (
            lf.with_columns([
                pl.col("date").str.slice(11, 2).cast(pl.Int32).alias("hour"),
                (pl.col("diesel").is_not_null() & pl.col("e10").is_not_null() & (pl.col("diesel") >= pl.col("e10"))).alias("anomaly")
            ])
            .group_by("hour")
            .agg([pl.len().alias("updates"), pl.col("anomaly").sum().alias("anomalies")])
            .with_columns((pl.col("anomalies") / pl.col("updates")).alias("anomaly_rate"))
            .sort("hour")
            .collect()
        )
        hourly.write_parquet(cache_path, compression="zstd", compression_level=19)
        
        del lf, hourly
        gc.collect()
        
    print("Hourly anomaly rates precomputed.")


def plot_anomaly_rate_by_hour(parquet_base: Path, year: int, anomalies_path: Path | None = None) -> go.Figure:
    """
    Plot anomaly rate (diesel >= e10) by hour of day for a given year
    Loads from precomputed cache in anomalies_path if available.
    """
    cache_path = anomalies_path / f"hourly_anomaly_rate_{year}.parquet" if anomalies_path else None
    if cache_path and cache_path.exists():
        hourly = pl.read_parquet(cache_path)
    else:
        lf = pl.scan_parquet(str(parquet_base / str(year) / "*.parquet"))
        hourly = (
            lf.with_columns([
                pl.col("date").str.slice(11, 2).cast(pl.Int32).alias("hour"),
                (
                    pl.col("diesel").is_not_null() &
                    pl.col("e10").is_not_null() &
                    (pl.col("diesel") >= pl.col("e10"))
                ).alias("anomaly")
            ])
            .group_by("hour")
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
        hovertemplate="Hour: %{x}<br>Anomaly Rate: %{y:.3f}<extra></extra>", # from Claude Code
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


def precompute_top_stations_map(stations_csv: Path, stations_parquet: Path, stats_cache: Path, anomalies_path: Path, min_updates: int = 50000) -> None:
    """
    Precompute joined station map data (filtered, sorted by anomaly_rate) and save to anomalies_path.
    The webapp can then slice Top N dynamically without reloading the large source files.
    Asked Claude Code for inspiration how to cache files for webapp
    """
    cache_path = anomalies_path / "top_stations_map.parquet"
    stations = load_stations(stations_csv, stations_parquet)
    station_stats = pl.read_parquet(stats_cache).filter(pl.col("updates") >= min_updates)
    joined = (
        station_stats
        .sort("anomaly_rate", descending=True)
        .join(stations, left_on="station_uuid", right_on="uuid", how="left")
        .with_columns(
            (pl.col("brand").fill_null("Unknown") + " " + pl.col("city").fill_null("")).alias("name")
        )
        .select(["station_uuid", "name", "latitude", "longitude", "anomalies", "updates", "anomaly_rate"])
    )
    joined.write_parquet(cache_path, compression="zstd", compression_level=19)


def plot_top_stations_map(stations_csv: Path, stations_parquet: Path, stats_cache: Path, start_year: int, end_year: int, top_n: int = 100, min_updates: int = 50000, anomalies_path: Path | None = None) -> go.Figure:
    """
    Load station data and stats, then plot the top N stations by anomaly rate.
    Loads from precomputed cache in anomalies_path if available.
    """
    cache_path = anomalies_path / "top_stations_map.parquet" if anomalies_path else None

    if cache_path and cache_path.exists():
        map_df = pl.read_parquet(cache_path).head(top_n)
    else:
        stations = load_stations(stations_csv, stations_parquet)
        station_stats = pl.read_parquet(stats_cache).filter(pl.col("updates") >= min_updates)
        map_df = prepare_station_map(station_stats, stations, top_n=top_n)

    station_data = map_df.to_pandas()

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


def plot_anomaly_duration_distribution(df_durations: pl.DataFrame) -> go.Figure:
    """
    Plot anomaly duration distribution as box plots per year (log scale).
    """
    duration_stats = (
        df_durations
        .with_columns((pl.col("duration_min") / 60).alias("duration_h"))
        .group_by("year")
        .agg([
            pl.col("duration_h").quantile(0.25).alias("q025"),
            pl.col("duration_h").median().alias("median"),
            pl.col("duration_h").quantile(0.75).alias("q075"),
            pl.col("duration_h").min().alias("min"),
            pl.col("duration_h").max().alias("max"),
        ])
        .sort("year")
    )

    fig = go.Figure()

    for row in duration_stats.iter_rows(named=True):
        fig.add_trace(go.Box(
            x=[row["year"]],
            q1=[round(row["q025"], 1)],
            median=[round(row["median"], 1)],
            q3=[round(row["q075"], 1)],
            lowerfence=[round(row["min"], 1)],
            upperfence=[round(row["max"], 1)],
            name=str(row["year"]),
            boxpoints=False,
        ))

    fig.update_layout(
        title="Anomaly Duration Distribution by Year",
        yaxis_type="log",
        yaxis_title="Duration (hours)"
    )

    return fig


def plot_median_anomaly_duration(df_durations: pl.DataFrame) -> go.Figure:
    """
    Plot median anomaly duration per year as a line chart.
    """
    year_stats = (
        df_durations.group_by("year")
        .agg((pl.col("duration_min").median() / 60).alias("median_duration_h"))
        .sort("year")
    )

    fig = px.line(
        year_stats.to_pandas(),
        x="year",
        y="median_duration_h",
        markers=True,
        title="Median Anomaly Duration by Year"
    )
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Median Duration (hours)"
    )
    return fig