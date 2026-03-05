import polars as pl

df = pl.read_json("pegelonline-kaub15min.json")

df = df.with_columns(
    pl.col("timestamp").str.to_datetime(format="%Y-%m-%dT%H:%M:%S%z")
)
df_hourly = df.filter(pl.col("timestamp").dt.minute() == 0)

df_hourly.write_csv("pegelonline-kaub-hourly.csv")