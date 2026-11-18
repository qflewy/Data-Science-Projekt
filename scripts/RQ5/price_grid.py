import polars as pl
from pathlib import Path

FUELS = ["e10", "diesel"]


def _month_frame(parquet_file: Path, fuels: list[str], price_range: tuple[float, float] | None) -> pl.DataFrame:
    """
    Reads one monthly price parquet and returns one row per station and price change.
    A price of 0 means the fuel is not offered (or not reported) and is set to null,
    with price_range also prices outside of it (implausible values).
    """
    def valid(f):
        return (pl.col(f) > 0) if price_range is None else pl.col(f).is_between(*price_range)

    return (
        pl.scan_parquet(parquet_file)
        .select([
            # local wall-clock time, the "+01"/"+02" offset is dropped on purpose
            pl.col("date").str.slice(0, 19).str.to_datetime("%Y-%m-%d %H:%M:%S").alias("timestamp"),
            "station_uuid",
            *[pl.when(valid(f)).then(pl.col(f)).otherwise(None).alias(f) for f in fuels],
        ])
        .drop_nulls("timestamp")
        .collect()
    )


def build_slot_means(parquet_base: str, year: int, months: list[int], every: str = "15m",
                     max_stale_hours: int = 72, fuels: list[str] = FUELS, groups: pl.DataFrame | None = None,
                     price_range: tuple[float, float] | None = None) -> pl.DataFrame:
    """
    Builds a time-weighted price series: for every station the price that is valid at the
    end of each slot (default 15 minutes) is carried forward until the next change.
    Afterwards the prices are averaged over all stations per slot, so every station counts
    once per slot regardless of how often it reports.

    Stations without any update for more than max_stale_hours are dropped from the slot,
    otherwise closed stations would be carried forward forever.

    Returns one row per slot with mean/median price per fuel, the number of stations and the
    share of stations with Diesel >= E10 (RQ5 anomaly, only if both fuels are included).
    With groups (columns station_uuid, group) one row per slot and group, e.g. brand vs. free stations.
    """
    base = Path(parquet_base) / str(year)
    carry = None  # last known state of every station from the previous month
    results = []

    for month in months:
        parquet_file = base / f"{year}-{month:02d}.parquet"
        if not parquet_file.exists():
            print(f"missing: {parquet_file}")
            continue

        df = _month_frame(parquet_file, fuels, price_range)
        month_start = pl.datetime(year, month, 1)
        month_end = df["timestamp"].max()

        # last price per station within each slot = price valid at the end of the slot
        per_slot = (
            df
            .with_columns(pl.col("timestamp").dt.truncate(every).alias("slot"))
            .sort(["station_uuid", "timestamp"])
            .group_by(["station_uuid", "slot"], maintain_order=True)
            .agg([pl.col(f).drop_nulls().last() for f in fuels] + [pl.col("timestamp").last().alias("last_seen")])
        )

        # anchor rows at the month start (with the carried state) and at the month end,
        # so that upsample covers the full month for every station
        anchors_start = (
            carry.with_columns(month_start.alias("slot"))
            if carry is not None
            else per_slot.group_by("station_uuid").agg(pl.col("slot").min()).with_columns(
                [pl.lit(None, pl.Float64).alias(f) for f in fuels] + [pl.lit(None, pl.Datetime("us")).alias("last_seen")]
            )
        )
        anchors_end = (
            pl.concat([per_slot.select("station_uuid"), anchors_start.select("station_uuid")])
            .unique()
            .with_columns(pl.lit(month_end).dt.truncate(every).alias("slot"))
            .with_columns([pl.lit(None, pl.Float64).alias(f) for f in fuels] + [pl.lit(None, pl.Datetime("us")).alias("last_seen")])
        )

        cols = ["station_uuid", "slot", *fuels, "last_seen"]
        stacked = (
            pl.concat([
                anchors_start.select(cols).with_columns(pl.lit(0).alias("prio")),
                per_slot.select(cols).with_columns(pl.lit(1).alias("prio")),
                anchors_end.select(cols).with_columns(pl.lit(-1).alias("prio")),
            ], how="vertical_relaxed")
            .with_columns(pl.col("slot").cast(pl.Datetime("us")), pl.col("last_seen").cast(pl.Datetime("us")))
            # a real observation beats an anchor in the same slot
            .sort(["station_uuid", "slot", "prio"])
            .group_by(["station_uuid", "slot"], maintain_order=True)
            .agg([pl.col(c).drop_nulls().last() for c in [*fuels, "last_seen"]])
            .sort(["station_uuid", "slot"])
        )

        grid = (
            stacked
            .upsample(time_column="slot", every=every, group_by="station_uuid")
            .with_columns(pl.col("station_uuid").forward_fill())
            .with_columns([pl.col(c).forward_fill().over("station_uuid") for c in [*fuels, "last_seen"]])
            .filter((pl.col("slot") - pl.col("last_seen")) <= pl.duration(hours=max_stale_hours))
        )

        carry = (
            grid.group_by("station_uuid").agg([pl.col(c).last() for c in [*fuels, "last_seen"]])
        )

        aggs = (
            [pl.col(f).mean().alias(f"{f}_mean") for f in fuels]
            + [pl.col(f).median().alias(f"{f}_median") for f in fuels]
            + [pl.col(f).count().alias(f"{f}_stations") for f in fuels]
        )
        if "diesel" in fuels and "e10" in fuels:
            # RQ5 anomaly: share of stations offering both fuels where Diesel >= E10
            aggs += [(pl.col("diesel") >= pl.col("e10")).mean().alias("anomaly_share"),
                     (pl.col("diesel") - pl.col("e10")).mean().alias("diesel_minus_e10")]

        grid = grid.filter(pl.col("slot") >= month_start)
        keys = ["slot"]
        if groups is not None:
            grid = grid.join(groups, on="station_uuid", how="inner")
            keys = ["slot", "group"]
        month_means = grid.group_by(keys).agg(aggs).sort(keys)
        results.append(month_means)
        print(f"{year}-{month:02d}: {month_means.height} rows, ~{month_means[f'{fuels[0]}_stations'].median():.0f} stations")
        del df, per_slot, stacked, grid

    return pl.concat(results).sort(["slot", "group"] if groups is not None else "slot")


if __name__ == "__main__":
    slots = build_slot_means("/home/marvin/tankerkoenig-data/prices_parquet", 2026, list(range(1, 12)))
    slots.write_parquet("/home/marvin/tankerkoenig-data/derived/2026_slot_means_15m.parquet")
