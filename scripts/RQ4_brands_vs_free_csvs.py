import polars as pl
from pathlib import Path

stations_parquet = Path("d:/Tankdaten/stations/stations/stations.parquet")
autobahn_csv = Path("d:/Tankdaten/stations/stations/autobahn_stations.csv")
prices_dir = Path("d:/Tankdaten/prices_parquet")
output_dir = Path("d:/Tankdaten/rq_brand_vs_free")

brands = ["ARAL", "SHELL", "JET", "TOTAL", "ESSO", "AVIA", "TOTAL ENERGIES", "HEM", "HOYER", "ORLEN", "Q1", "ORLEN", "STAR", "RAIFFEISEN", "AGIP", "OMV", "OIL!", "WESTFALEN", "ENI"]

# Known brand variants in the data → mapped to their canonical brand name above.
brand_aliasses: dict[str, str] = {
    "AVIA SCHMIDT":                    "AVIA",
    "AVIA XPRESS":                     "AVIA",
    "AVIA XPRESS AUTOMATENTANKSTELLE": "AVIA",
    "SHELL BASEDOW":                   "SHELL",
    "HOYER BEI DODENHOF":              "HOYER",
    "HOYER TANK-TREFF LINDAU":         "HOYER",
    "WESTFALEN TANKSTELLE":            "WESTFALEN",
    "WESTFALEN-TANKSTELLE":            "WESTFALEN",
}

def load_and_classify_stations(parquet_path: Path, brands, autobahn_csv: Path) -> pl.DataFrame:
    """
    Returns a DataFrame with columns:
        uuid, brand_normalized, station_type ("brand_station" | "free_station")
    Highway (Autobahn) stations are excluded entirely.
    """
    df = pl.read_parquet(parquet_path)

    # Exclude highway stations
    autobahn_uuids = pl.read_csv(autobahn_csv, columns=["uuid"])["uuid"]
    df = df.filter(~pl.col("uuid").is_in(autobahn_uuids))

    # Normalize: fold to uppercase
    df = df.with_columns(
        pl.col("brand")
        .str.strip_chars()
        .str.to_uppercase()
        .alias("brand_normalized")
    )

    # Apply aliases: e.g. "AVIA XPRESS" → "AVIA"
    df = df.with_columns(
        pl.col("brand_normalized")
        .replace(brand_aliasses)
        .alias("brand_normalized")
    )

    # Classify: exact match against brands list → "brand_station", everything else → "free_station"
    df = df.with_columns(
        pl.when(
            pl.col("brand_normalized").is_not_null()
            & pl.col("brand_normalized").is_in(brands)
        )
        .then(pl.lit("brand_station"))
        .otherwise(pl.lit("free_station"))
        .alias("station_type")
    )

    return df.select(["uuid", "brand_normalized", "station_type"])


def _agg_month(df_month: pl.DataFrame, group_cols: list[str]) -> pl.DataFrame:
    """
    Aggregate one month's data. Stores sum+count for each fuel so that
    means can be combined correctly across months later.
    Prices outside [0.5, 4.0] are treated as invalid and excluded.
    """
    fuels = ["diesel", "e5", "e10"]

    # Null out implausible prices in-place so they don't enter sums/counts
    df_month = df_month.with_columns([
        pl.when(pl.col(f).is_between(0.5, 4.0)).then(pl.col(f)).otherwise(None).alias(f)
        for f in fuels
    ])

    return (
        df_month
        .group_by(group_cols)
        .agg(
            [pl.col(f).sum().alias(f"{f}_sum") for f in fuels]
            + [pl.col(f).count().alias(f"{f}_count") for f in fuels]
            + [pl.len().alias("price_update_count")]
        )
    )


def _combine_and_finalize(partials: list[pl.DataFrame], group_cols: list[str]) -> pl.DataFrame:
    """
    Concat all monthly partial aggregates, re-group, and compute final means
    from (sum, count) pairs.
    """
    fuels = ["diesel", "e5", "e10"]

    combined = (
        pl.concat(partials)
        .group_by(group_cols)
        .agg(
            [pl.col(f"{f}_sum").sum() for f in fuels]
            + [pl.col(f"{f}_count").sum() for f in fuels]
            + [pl.col("price_update_count").sum()]
        )
        .with_columns([
            (pl.col(f"{f}_sum") / pl.col(f"{f}_count")).alias(f"{f}_mean")
            for f in fuels
        ])
        .drop([f"{f}_sum" for f in fuels] + [f"{f}_count" for f in fuels])
        .sort(group_cols)
    )
    return combined


def compute_price_stats(prices_dir: Path, station_classification: pl.DataFrame, output_dir: Path,) -> None:
    """
    Processes one parquet file at a time to keep RAM usage low.
    Each file is read, joined, aggregated (tiny result), then discarded.
    Final means are computed by combining (sum, count) pairs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(prices_dir.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {prices_dir}")

    partials_type:  list[pl.DataFrame] = []
    partials_brand: list[pl.DataFrame] = []

    for pf in parquet_files:
        # Load only the needed columns
        df = pl.read_parquet(pf, columns=["station_uuid", "date", "diesel", "e5", "e10"])

        # Join with station classification (small lookup table, stays in RAM)
        df = df.join(station_classification, left_on="station_uuid", right_on="uuid", how="inner")

        # Extract year
        df = df.with_columns(
            pl.col("date").str.slice(0, 4).cast(pl.Int16).alias("year")
        )

        # Aggregate by station_type × year for all stations
        partials_type.append(_agg_month(df, ["station_type", "year"]))

        # Aggregate by brand × year for brand stations only
        df_brand = df.filter(pl.col("station_type") == "brand_station")
        partials_brand.append(_agg_month(df_brand, ["brand_normalized", "year"]))

        print(f"{df.height:,} rows")
        del df, df_brand  # free RAM immediately

    print("\nCombining monthly results...")

    df_stats = _combine_and_finalize(partials_type, ["station_type", "year"])
    out_file = output_dir / "brand_vs_free_yearly_stats.csv"
    df_stats.write_csv(out_file)
    print(f"Saved yearly stats: {out_file}")
    print(df_stats.head(10))

    df_brands = _combine_and_finalize(partials_brand, ["brand_normalized", "year"])
    brand_file = output_dir / "per_brand_yearly_stats.csv"
    df_brands.write_csv(brand_file)
    print(f"Saved per-brand stats: {brand_file}")
    print(df_brands.head(10))


if __name__ == "__main__":
    station_df = load_and_classify_stations(stations_parquet, brands, autobahn_csv)
    compute_price_stats(prices_dir, station_df, output_dir)
    print("\nDone.")