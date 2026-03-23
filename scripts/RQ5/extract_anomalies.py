import polars as pl
from pathlib import Path

def extract_anomalies_from_parquet(year: int = None, month: int = None, base_path: str = None, output_dir: str = None):
    """
    Extracts anomalies from Parquet files:
    anomaly:
        - Diesel Price >= E10 Price
    """
    if not base_path:
        base_path = "D:/Tankdaten/prices_parquet"
    if not output_dir:
        output_dir = "D:/Tankdaten/anomalies"
    
    year_path = Path(base_path) / str(year)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not year_path.exists():
        print(f"Year {year_path} not found!") #nice to have if you have a typo in the year (2034 instead of 2024)
        return
    
    all_results = []
    months_processed = 0
    total_anomalies = 0

    if month:
        month_pattern = f"{year}-{month:02d}.parquet"  # :02d zero-pads single digits, e.g. month 3 → "03"
        parquet_files = [year_path / month_pattern] if (year_path / month_pattern).exists() else []  # wrap in list so the for-loop below works the same as for a full year
    else:
        parquet_files = sorted(year_path.glob(f"{year}-*.parquet"))
    
    for parquet_file in parquet_files:
        month_name = parquet_file.stem.split("-")[1]  # .stem strips the .parquet extension, split gives ["2022", "03"] → take index 1 for the month
        try:
            df = pl.read_parquet(parquet_file)
            
            anomalies = df.filter(
                pl.col("diesel").is_not_null() & # Filtered for stations who have both Diesel and E10
                pl.col("e10").is_not_null() &
                (
                    (pl.col("diesel") == pl.col("e10")) |
                    (pl.col("diesel") > pl.col("e10"))
                )
            ).drop(["e5", "e5change"])  # Drop E5
            
            if anomalies.height > 0:  # .height for number of rows
                anomalies = anomalies.sort("date")
                
                monthly_output = output_path / f"{year}_{month_name}_anomalies.parquet"
                anomalies.write_parquet(monthly_output)
                all_results.append(anomalies)
                months_processed += 1
                total_anomalies += anomalies.height
                print(f"{anomalies.height} Anomalies found")
            else:
                print("no anomalies")
                months_processed += 1
        
        except Exception as e:
            print(f"Failure: {e}")

    if all_results:
        final_df = pl.concat(all_results)
        final_df = final_df.sort("date")
        
        if month:
            output_file = output_path / f"{year}_{month:02d}_anomalies.parquet" # if a month is processed, save as a monthly file
        else:
            output_file = output_path / f"{year}_all_anomalies.parquet" # if a whole year is processed, save as a yearly file
        
        final_df.write_parquet(output_file)
        
        print(f"{year} finished")
    else:
        print(f"No Anomalies found in {year}.")


if __name__ == "__main__":
    parquet_base = "D:/Tankdaten/prices_parquet"
    output = "D:/Tankdaten/anomalies"
#    for year in range(2023, 2027):
       #extract_anomalies_from_parquet(year=year, base_path=parquet_base, output_dir=output)
    extract_anomalies_from_parquet(year=2022, base_path=parquet_base, output_dir=output)