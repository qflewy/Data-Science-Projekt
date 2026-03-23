# modified from build_station_daily_last_by_month.py to calculate mean and
# median instead of last price. Modified with the help of Copilot.

from pathlib import Path
import polars as pl


def build_station_daily_by_month(data_root, derived_root, start_year, end_year) -> None:
    """Generate daily mean/median price files for each station by month.

    Parameters
    ----------
    data_root : Path | str
        Root folder containing subfolders "prices/{year}/{month}" with CSV price files.
    derived_root : Path | str
        Output root where "station_daily_last_by_month/{year}" subfolders are written.
    start_year : int, optional
        First year (inclusive) to process. Defaults to 2014.
    end_year : int, optional
        Last year (inclusive) to process. Defaults to 2026.
    """
    DATA_ROOT = Path(data_root)
    DERIVED_ROOT = Path(derived_root)

    for YEAR in range(start_year, end_year + 1):

        IN_YEAR_DIR = DATA_ROOT / "prices" / str(YEAR)
        OUT_YEAR_DIR = DERIVED_ROOT / "station_daily_mean_and_median_by_month" / str(YEAR)
        OUT_YEAR_DIR.mkdir(parents=True, exist_ok=True)

        # LOOP MONTHS
        for month in range(1, 13):
            mm = f"{month:02d}"
            in_glob = str(IN_YEAR_DIR / mm / "*-prices.csv")
            out_file = OUT_YEAR_DIR / f"{YEAR}-{mm}.parquet"

            # Skip if file already exists (also helps to restart script if it failed somewhere in the middle):
            if out_file.exists():
                print("Skip (exists):", out_file)
                continue

            print(f"[{YEAR}-{mm}] reading:", in_glob)

            lf = pl.scan_csv(in_glob, infer_schema_length=0, ignore_errors=True)

            try:
                _ = lf.collect_schema()
            except Exception:
                print(f"[{YEAR}-{mm}] no files / unreadable -> skip")
                continue

            # this is to fix the date parsing, because some dates have +01/+02 instead of +0100/+0200, which causes strptime to fail.
            # We also extract the day (date without time) for later grouping.
            # ChatGPT used for generating the next section:
            lf = (
                lf.with_columns(
                    pl.col("date")
                    .str.replace(r"([+-]\d{2})$", "${1}00")
                    .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%z", strict=True)
                    .alias("dt")
                )
                .with_columns(
                    pl.col("dt").dt.date().alias("day")
                )
                .with_columns([
                    pl.col("diesel").cast(pl.Float64),
                    pl.col("e5").cast(pl.Float64),
                    pl.col("e10").cast(pl.Float64),
                ])
                .with_columns([
                    pl.when(pl.col("diesel") <= 0).then(None).otherwise(pl.col("diesel")).alias("diesel"),
                    pl.when(pl.col("e5") <= 0).then(None).otherwise(pl.col("e5")).alias("e5"),
                    pl.when(pl.col("e10") <= 0).then(None).otherwise(pl.col("e10")).alias("e10"),
                ])
            )

            # Sorting by month, station and day, then taking the mean and median 
            # price of each day for each station
            # (calculated from all prices of the day for each station).
            station_daily_month = (
                lf.sort("dt")
                .group_by(["station_uuid", "day"])
                .agg([
                    pl.col("diesel").drop_nulls().mean().alias("diesel_mean"),
                    pl.col("diesel").drop_nulls().median().alias("diesel_median"),
                    pl.col("e5").drop_nulls().mean().alias("e5_mean"),
                    pl.col("e5").drop_nulls().median().alias("e5_median"),
                    pl.col("e10").drop_nulls().mean().alias("e10_mean"),
                    pl.col("e10").drop_nulls().median().alias("e10_median"),
                    pl.len().alias("n_events"),
                ])
                .sort(["day", "station_uuid"])
            )

            station_daily_month.sink_parquet(out_file)
            print("Wrote:", out_file)

        print(f"[{YEAR}] DONE")

    print("All done. Everything went smoothly, FINALLY!!!")


if __name__ == "__main__":
    # allow running as a script with default paths
    build_station_daily_by_month(
        r"C:\Users\Bjarne\Desktop\Uni\Data Science Projekt\TankKoenigData",
        r"C:\Users\Bjarne\Desktop\Uni\Data Science Projekt\PersonalTesting\dbscan\derived",
    )