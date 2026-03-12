from __future__ import annotations
from pathlib import Path
import polars as pl


def build_price_change_events_month(
    year: int,
    month: int,
    input_root: Path = Path(r"D:/data/tankerkoenig-data/prices"),
    output_root: Path = Path(r"D:/data/derived/price_change_events"),
) -> Path:

    mm = f"{month:02d}"
    month_dir = input_root / str(year) / mm
    pattern = str(month_dir / "*-prices.csv")
    print("Reading:", pattern)
    lf = (
        pl.scan_csv(pattern)
        .rename({"station_uuid": "station_id"})
        .select(["date", "station_id", "diesel", "e5", "e10"])
        .with_columns(
            pl.col("date")
            .str.replace(r"([+-]\d{2})$", "${1}00")
            .str.to_datetime(format="%Y-%m-%d %H:%M:%S%z", strict=False)
            .alias("timestamp")
        )
    )
    prices = (
        lf.unpivot(
            index=["timestamp", "station_id"],
            on=["diesel", "e5", "e10"],
            variable_name="fuel_type",
            value_name="price",
        )
        .filter(pl.col("price") > 0)
        .sort(["station_id", "fuel_type", "timestamp"])
        .with_columns(
            pl.col("price")
            .shift(1)
            .over(["station_id", "fuel_type"])
            .alias("prev_price")
        )
        .with_columns(
            (pl.col("price") - pl.col("prev_price")).round(3).alias("price_change")
        )
        .filter(pl.col("price_change").is_not_null())
        .filter(pl.col("price_change") != 0)
        .with_columns(
            [
                pl.when(pl.col("price_change") > 0)
                .then(pl.lit("incr"))
                .otherwise(pl.lit("decr"))
                .alias("change_type"),
                pl.col("timestamp").dt.date().alias("date"),
                pl.col("timestamp").dt.hour().alias("hour"),
                pl.col("timestamp").dt.weekday().alias("weekday"),
            ]
        )
        .select(
            [
                "station_id",
                "timestamp",
                "date",
                "hour",
                "weekday",
                "fuel_type",
                "price_change",
                "change_type",
            ]
        )
    )
    output_root.mkdir(parents=True, exist_ok=True)
    out_path = output_root / f"price_change_events_{year}_{mm}.parquet"
    prices.sink_parquet(out_path)

    return out_path

# ---------------------------------------------

def build_daily_price_range_month(
    year: int,
    month: int,
    input_root: Path = Path(r"D:/data/tankerkoenig-data/prices"),
    output_root: Path = Path(r"D:/data/derived/daily_price_range"),
) -> Path:
    """
    Build monthly daily_price_range parquet from Tankerkönig raw daily CSV files.
    Output columns:
        - station_id
        - date
        - fuel_type
        - daily_max
        - daily_min
        - daily_range
    """

    mm = f"{month:02d}"
    month_dir = input_root / str(year) / mm
    if not month_dir.exists():
        raise FileNotFoundError(f"Month folder not found: {month_dir}")

    pattern = str(month_dir / "*-prices.csv")
    print(f"Reading: {pattern}")
    lf = (
        pl.scan_csv(
            pattern,
            has_header=True,
            infer_schema_length=1000,
            try_parse_dates=False,
        )
        .rename({"station_uuid": "station_id"})
        .select(["date", "station_id", "diesel", "e5", "e10"])
        .with_columns(
            pl.col("date")
            .str.replace(r"([+-]\d{2})$", "${1}00")
            .str.to_datetime(format="%Y-%m-%d %H:%M:%S%z", strict=False)
            .dt.convert_time_zone("Europe/Berlin")
            .alias("timestamp")
        )
    )
    daily_ranges = (
        lf.unpivot(
            index=["timestamp", "station_id"],
            on=["diesel", "e5", "e10"],
            variable_name="fuel_type",
            value_name="price",
        )
        .filter(pl.col("price") > 0)
        .with_columns(
            pl.col("timestamp").dt.date().alias("date")
        )
        .group_by(["station_id", "date", "fuel_type"])
        .agg(
            [
                pl.col("price").max().alias("daily_max"),
                pl.col("price").min().alias("daily_min"),
            ]
        )
        .with_columns(
            (pl.col("daily_max") - pl.col("daily_min")).round(3).alias("daily_range")
        )
        .select(
            [
                "station_id",
                "date",
                "fuel_type",
                "daily_max",
                "daily_min",
                "daily_range",
            ]
        )
        .sort(["date", "station_id", "fuel_type"])
    )
    output_root.mkdir(parents=True, exist_ok=True)
    out_path = output_root / f"daily_price_range_{year}_{mm}.parquet"

    daily_ranges.sink_parquet(out_path)
    return out_path

# ---------------------------------------------

def combine_daily_price_range_files(
    input_root: Path = Path(r"D:/data/derived/daily_price_range"),
    output_root: Path = Path(r"D:/data/derived/daily_price_range_combined"),
    mode: str = "year",
) -> None:
    """
    Combine monthly daily_price_range parquet files.

    mode:
        - "year": combine monthly files into one parquet per year
        - "all": combine all monthly files into one parquet for all years
    """

    output_root.mkdir(parents=True, exist_ok=True)

    monthly_files = sorted(input_root.glob("daily_price_range_*.parquet"))
    if not monthly_files:
        raise FileNotFoundError(f"No parquet files found in {input_root}")

    if mode == "year":
        files_by_year: dict[str, list[Path]] = {}

        for file in monthly_files:
            # expected: daily_price_range_2025_01.parquet
            parts = file.stem.split("_")
            year = parts[-2]
            files_by_year.setdefault(year, []).append(file)

        for year, files in files_by_year.items():
            print(f"Combining year {year} ({len(files)} files)...")

            lf = pl.scan_parquet([str(f) for f in files])
            out_path = output_root / f"daily_price_range_{year}.parquet"
            lf.sink_parquet(out_path)

            print(f"[OK] Saved: {out_path}")

    elif mode == "all":
        print(f"Combining all files ({len(monthly_files)} files)...")

        lf = pl.scan_parquet([str(f) for f in monthly_files])
        out_path = output_root / "daily_price_range_all.parquet"
        lf.sink_parquet(out_path)

        print(f"[OK] Saved: {out_path}")

    else:
        raise ValueError("mode must be either 'year' or 'all'")
    
 # ---------------------------------------------

def build_hourly_price_changes_month(
    year: int,
    month: int,
    input_root: Path = Path(r"D:/data/derived/price_change_events"),
    output_root: Path = Path(r"D:/data/derived/hourly_price_changes"),
) -> Path:
    """
    Build monthly hourly_price_changes parquet from monthly price_change_events parquet.

    Input:
        price_change_events_{year}_{month:02d}.parquet

    Output columns:
        - date
        - hour
        - weekday
        - fuel_type
        - number_of_price_changes
        - increase
        - decrease
    """
    mm = f"{month:02d}"
    in_path = input_root / f"price_change_events_{year}_{mm}.parquet"
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")
    lf = pl.scan_parquet(in_path)
    hourly = (
        lf.group_by(["date", "hour", "weekday", "fuel_type"])
        .agg(
            [
                pl.len().alias("number_of_price_changes"),
                (pl.col("change_type") == "incr").sum().alias("increase"),
                (pl.col("change_type") == "decr").sum().alias("decrease"),
            ]
        )
        .sort(["date", "hour", "fuel_type"])
    )

    output_root.mkdir(parents=True, exist_ok=True)
    out_path = output_root / f"hourly_price_changes_{year}_{mm}.parquet"

    hourly.sink_parquet(out_path)
    return out_path

# ---------------------------------------------

def combine_hourly_price_changes_all(
    input_root: Path = Path(r"D:/data/derived/hourly_price_changes"),
    output_file: Path = Path(r"D:/data/derived/hourly_price_changes_all.parquet"),
):

    files = sorted(input_root.glob("hourly_price_changes_*.parquet"))

    if not files:
        raise FileNotFoundError("No hourly_price_changes parquet files found")

    print(f"Combining {len(files)} files...")

    lf = pl.scan_parquet([str(f) for f in files])

    lf.sink_parquet(output_file)

    print(f"[OK] Saved combined file: {output_file}")

# ---------------------------------------------

def build_monthly_price_observations(
    year: int,
    month: int,
    data_root: str | Path = r"D:/data/tankerkoenig-data",
    out_root: str | Path = r"D:/data/derived/station_price_observations",
    keep_station_meta: bool = True,
) -> Path:
    """
    Erstellt eine monatliche Analyse-Datei für Tankzeit-/Preisanalysen.

    Input:
      - prices/YYYY/MM/*-prices.csv
      - stations/YYYY/MM/*-stations.csv

    Output:
      - station_price_observations_YYYY_MM.parquet

    Zielspalten:
      station_id, timestamp, date, hour, weekday,
      fuel_type, price, price_changed,
      optional: brand, city, post_code, latitude, longitude
    """

    data_root = Path(data_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    mm = f"{month:02d}"

    prices_glob = str(data_root / "prices" / str(year) / mm / "*-prices.csv")
    stations_glob = str(data_root / "stations" / str(year) / mm / "*-stations.csv")

    out_file = out_root / f"station_price_observations_{year}_{mm}.parquet"

    prices_lf = (
        pl.scan_csv(
            prices_glob,
            infer_schema_length=2000,
            try_parse_dates=False,
            ignore_errors=False,
        )
        .select([
            pl.col("date"),
            pl.col("station_uuid"),
            pl.col("diesel"),
            pl.col("e5"),
            pl.col("e10"),
            pl.col("dieselchange"),
            pl.col("e5change"),
            pl.col("e10change"),
        ])
        .rename({"station_uuid": "station_id"})
    )

    prices_lf = prices_lf.with_columns(
        pl.col("date")
        .str.replace(r"\+\d+$", "")
        .str.to_datetime("%Y-%m-%d %H:%M:%S")
        .alias("timestamp")
    )

    prices_long = (
        prices_lf
        .select([
            "station_id",
            "timestamp",
            pl.col("diesel").alias("diesel"),
            pl.col("e5").alias("e5"),
            pl.col("e10").alias("e10"),
        ])
        .unpivot(
            index=["station_id", "timestamp"],
            on=["diesel", "e5", "e10"],
            variable_name="fuel_type",
            value_name="price",
        )
    )

    changes_long = (
        prices_lf
        .select([
            "station_id",
            "timestamp",
            pl.col("dieselchange").alias("diesel"),
            pl.col("e5change").alias("e5"),
            pl.col("e10change").alias("e10"),
        ])
        .unpivot(
            index=["station_id", "timestamp"],
            on=["diesel", "e5", "e10"],
            variable_name="fuel_type",
            value_name="price_changed",
        )
    )

    obs_lf = (
        prices_long
        .join(
            changes_long,
            on=["station_id", "timestamp", "fuel_type"],
            how="left",
        )

        .filter(
            pl.col("price").is_not_null() &
            (pl.col("price") > 0)
        )
    )

    obs_lf = obs_lf.with_columns([
        pl.col("timestamp").dt.date().alias("date"),
        pl.col("timestamp").dt.hour().alias("hour"),
        (pl.col("timestamp").dt.weekday() - 1).alias("weekday"),
    ])

    if keep_station_meta:
        stations_lf = (
            pl.scan_csv(
                stations_glob,
                infer_schema_length=2000,
                try_parse_dates=False,
                ignore_errors=False,
            )
            .select([
                pl.col("uuid").alias("station_id"),
                "name",
                "brand",
                "street",
                "house_number",
                "post_code",
                "city",
                "latitude",
                "longitude",
            ])
        )

        stations_df = (
            stations_lf
            .collect()
            .unique(subset=["station_id"], keep="last")
            .lazy()
        )

        obs_lf = obs_lf.join(stations_df, on="station_id", how="left")

    base_cols = [
        "station_id",
        "timestamp",
        "date",
        "hour",
        "weekday",
        "fuel_type",
        "price",
        "price_changed",
    ]

    meta_cols = [
        "name",
        "brand",
        "street",
        "house_number",
        "post_code",
        "city",
        "latitude",
        "longitude",
    ]

    final_cols = base_cols + ([c for c in meta_cols] if keep_station_meta else [])

    obs_lf = obs_lf.select(final_cols)

    obs_df = obs_lf.collect(engine="streaming").sort(["station_id", "fuel_type", "timestamp"])
    obs_df.write_parquet(out_file)

    print(f"Gespeichert: {out_file}")
    print(obs_df.shape)
    print(obs_df.head())

    return out_file


def build_range(
    years: list[int],
    months: list[int] | None = None,
    data_root: str | Path = r"D:/data/tankerkoenig-data",
    out_root: str | Path = r"D:/data/derived/station_price_observations",
    keep_station_meta: bool = True,
    skip_existing: bool = True,
) -> None:
    """
    Baut mehrere Monatsdateien.
    """
    if months is None:
        months = list(range(1, 13))

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for year in years:
        for month in months:
            mm = f"{month:02d}"
            out_file = out_root / f"station_price_observations_{year}_{mm}.parquet"

            if skip_existing and out_file.exists():
                print(f"Skip (exists): {out_file}")
                continue

            try:
                build_monthly_price_observations(
                    year=year,
                    month=month,
                    data_root=data_root,
                    out_root=out_root,
                    keep_station_meta=keep_station_meta,
                )
            except Exception as e:
                print(f"Fehler bei {year}-{mm}: {e}")

# ------------------------------
# Execution
# ------------------------------
if __name__ == "__main__":

    # for year in range(2015, 2025):
    #     for month in range(1, 13):
    #         out_file = build_daily_price_range_month(year, month)
    #         print(f"Saved to: {out_file}")

    #combine_daily_price_range_files(mode="year")

    # for year in range(2015, 2025):
    #     for month in range(1, 13):
    #         out_file = build_hourly_price_changes_month(year, month)
    #         print(f"Saved to: {out_file}")
    
    #combine_hourly_price_changes_all()

    build_monthly_price_observations(2026, 2)