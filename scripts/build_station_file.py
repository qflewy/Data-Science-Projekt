from pathlib import Path
import polars as pl

#builds stations file including the post tankrabatt stations
def build_stations_csv():
    goalpath = Path(r'/Users/sebastian/data-science-projekt/tankerkoenig_data/stations/')
    stationsfile = goalpath/ 'stations.csv'
    roots = [
        goalpath / '2026/03',
        goalpath / '2026/04',
        goalpath / '2026/05',
        goalpath / '2026/06',
        goalpath / '2026/07'
    ]

    schema_overrides = {
        "uuid": pl.String,
        "name": pl.String,
        "brand": pl.String,
        "street": pl.String,
        "house_number": pl.String,
        "post_code": pl.String,
        "city": pl.String,
        "latitude": pl.Float64,
        "longitude": pl.Float64,
    }

    files = [
        file 
        for root in roots
        for file in root.rglob("*.csv")
    ]
    files.append(stationsfile)

    frames = [
        pl.scan_csv(file, schema_overrides=schema_overrides).select(schema_overrides.keys())
        for file in files
    ]

    results = (
        pl.concat(frames)
        .unique(subset=["uuid"])
    )

    results.sink_csv(goalpath / 'stations_post_TR.csv')

if __name__ == "__main__":
    df1 = pl.scan_csv('/Users/sebastian/data-science-projekt/tankerkoenig_data/stations/autobahn_stations.csv', ignore_errors=True)
    df2 = pl.scan_csv('/Users/sebastian/data-science-projekt/tankerkoenig_data/stations/autobahn_stations_post_TR.csv', ignore_errors=True)

    print(df1.describe())
    print(df1.count())

    print(df2.describe())
    print(df2.count())
    
