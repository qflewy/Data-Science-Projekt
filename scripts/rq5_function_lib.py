from pathlib import Path
import polars as pl
import pandas as pd

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
    Count total price updates per station.
    Iterates monthly parquet files to stay RAM efficient.
    """

    counts = {}

    for year in range(start_year, end_year):
        year_path = prices_path / str(year)

        for file in sorted(year_path.glob("*.parquet")):

            df = pl.read_parquet(file, columns=["station_uuid"])

            grouped = (
                df
                .group_by("station_uuid")
                .agg(pl.len().alias("updates"))
            )

            for uuid, n in grouped.iter_rows():
                counts[uuid] = counts.get(uuid, 0) + n

    return pl.DataFrame({
        "station_uuid": list(counts.keys()),
        "updates": list(counts.values())
    })


def anomaly_rate(anomalies: pl.DataFrame, updates: pl.DataFrame) -> pl.DataFrame:
    """
    Compute anomaly rate per station.
    """

    return (
        anomalies
        .join(updates, on="station_uuid", how="left")
        .with_columns(
            (pl.col("anomalies") / pl.col("updates")).alias("anomaly_rate")
        )
    )


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