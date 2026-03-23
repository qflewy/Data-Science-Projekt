from pathlib import Path
import polars as pl

DERIVED_ROOT = Path(r"D:\data\derived")
for YEAR in range(2014, 2027):

    IN_GLOB = str(DERIVED_ROOT / "station_daily_last_by_month" / str(YEAR) / "*.parquet")
    OUT_DIR = DERIVED_ROOT / "national_daily_last"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_csv = OUT_DIR / f"national_daily_last_{YEAR}.csv"
    out_parq = OUT_DIR / f"national_daily_last_{YEAR}.parquet"

    lf = pl.scan_parquet(IN_GLOB)

    national = (
        lf.group_by("day")
        .agg([
            pl.col("diesel_last").median().alias("diesel_median_last"),
            pl.col("diesel_last").mean().alias("diesel_mean_last"),
            pl.col("e5_last").median().alias("e5_median_last"),
            pl.col("e5_last").mean().alias("e5_mean_last"),
            pl.col("e10_last").median().alias("e10_median_last"),
            pl.col("e10_last").mean().alias("e10_mean_last"),
            pl.len().alias("n_stations_seen"),
        ])
        .sort("day")
    )

    national.collect().write_csv(out_csv)
    national.sink_parquet(out_parq)

    print("Wrote:", out_csv)
    print("Wrote:", out_parq)

print("Done.")