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
    Count total updates and anomalies (diesel >= e10) per station.
    Only rows where both diesel > 0 and e10 > 0 are considered (excludes diesel-only stations).
    Processes one monthly file at a time to stay within RAM limits.
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

def anomaly_rate(updates: pl.DataFrame) -> pl.DataFrame:
    """
    Compute anomaly rate per station.
    Expects a DataFrame with station_uuid, updates, and anomalies columns
    as returned by total_updates().
    """

    return updates.with_columns(
        (pl.col("anomalies") / pl.col("updates")).alias("anomaly_rate")
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