import polars as pl

from pathlib import Path

def create_region_hourly_mean_median_prices(price_folder:Path, regions_path:Path, output_path:Path):
    
    
    # --- 1. Regionen-Tabelle vorbereiten ---
    
    region_dfs = [pl.read_csv(f).with_columns(pl.lit(f.stem.split("_")[-1]).alias("Region_Code")) for f in regions_path.glob("uuids_region_*.csv")]
    df_all_regions = pl.concat(region_dfs)

    # --- 2. Preisdaten lazily einlesen ---pro jahr einlesen
    year_folder = [p for p in price_folder.iterdir() if p.is_dir()]

    collected_results = []
    for year_path in year_folder:
        print(f"processing year: {year_path.name}...")
        
        lazy_prices = pl.scan_csv(f"{year_path}/*/*.csv")

        # --- 3. Der Bauplan für TAGE (und Stunden) ---
        results_lazy = (
            lazy_prices
            .join(df_all_regions.lazy(),
                left_on="station_uuid", #from price files
                right_on="uuid", #from region files
                how="inner")
            
            # Wir extrahieren den exakten Tag (die ersten 10 Zeichen)
        
            .with_columns(
                pl.col("date").str.slice(0, 10).alias("day"),
                pl.col("date").str.slice(11, 2).cast(pl.Int8).alias("hour")
            )
            
            # Jetzt gruppieren wir nach Region, Tag UND Stunde.
            .group_by(["Region_Code", "day", "hour"]) 
            
            # Durchschnitt und Median berechnen
            .agg([
                pl.col("diesel").mean().alias("diesel_mean"),
                pl.col("diesel").median().alias("diesel_median"),
                pl.col("e5").mean().alias("e5_mean"),
                pl.col("e5").median().alias("e5_median"),
                pl.col("e10").mean().alias("e10_mean"),
                pl.col("e10").median().alias("e10_median")
            ])
            
            # Alles chronologisch sauber sortieren
            .sort(["Region_Code", "day", "hour"])
            # Ausführung der Berechnung
            .collect(streaming=True)
        )

        collected_results.append(results_lazy)
        

    print("All years calculated. Putting together results.")
    # --- 5. In separate CSVs pro Region abspeichern ---
    df_processed = pl.concat(collected_results)
    
    #converting into datetime[μs, UTC]
    df_processed = df_processed.with_columns(
        (
            pl.col("day") + " " + 
            pl.col("hour").cast(pl.String).str.zfill(2) + ":00:00"
        )
        # Zwingt es in das Format datetime[μs, UTC]
        .str.to_datetime("%Y-%m-%d %H:%M:%S", time_unit="us", time_zone="UTC")
        .alias("timestamp_utc")
    ).drop(["day", "hour"]) # Die alten Spalten direkt wegwerfen, um Platz zu sparen

    # Zur Sicherheit nochmal nach Region und dem neuen Zeitstempel sortieren
    df_processed = df_processed.sort(["Region_Code", "timestamp_utc"])

    # Das riesige DataFrame wieder in Regionen aufteilen
    for df_region in df_processed.partition_by("Region_Code"):
        current_region = df_region["Region_Code"][0]
        file_path = output_path / f"mean_median_price_region_{current_region}.csv"
        
        # Region_Code-Spalte droppen (brauchen wir nicht mehr in der Datei) und speichern
        df_region.drop("Region_Code").write_csv(file_path)
        print(f"Created: {file_path} ({df_region.height} rows)")




