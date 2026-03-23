import polars as pl
from pathlib import Path

def convert_csvs_to_parquet(base_csv_path: str, base_parquet_path: str, year: int = None, month: int = None):
    """
    Converts all CSV files to Parquet files
    All CSVs of one month combined in one parquet file
    Structur: prices/YYYY/MM/*.csv -> prices_parquet/YYYY/YYYY-MM.parquet
    """
    
    base_csv = Path(base_csv_path)
    base_parquet = Path(base_parquet_path)
    
    base_parquet.mkdir(parents=True, exist_ok=True)
    
    total_months = 0
    total_rows = 0
    
    if year:
        years_to_process = [year]
    else:
        years_to_process = [int(y.name) for y in sorted(base_csv.iterdir()) if y.is_dir()]
    
    for year_val in years_to_process:
        year_dir = base_csv / str(year_val)
        if not year_dir.exists():
            continue
        
        year_parquet = base_parquet / str(year_val)
        year_parquet.mkdir(parents=True, exist_ok=True)
                
        if month:
            months_to_process = [month]
        else:
            months_to_process = range(1, 13)
        
        for month_val in months_to_process:
            month_str = f"{month_val:02d}"
            month_dir = year_dir / month_str
            
            if not month_dir.exists():
                continue
            
            out_file = year_parquet / f"{year_val}-{month_str}.parquet"
            
            if out_file.exists():
                continue
            
            in_glob = str(month_dir / "*-prices.csv")
            
            try:                
                lf = pl.scan_csv(in_glob, ignore_errors=True)
                result = lf.collect()
                
                if result.height == 0:
                    continue
                
                result.write_parquet(out_file)
                
                total_months += 1
                total_rows += result.height
                    
            except Exception as e:
                print(f"Fehler: {e}")
    
    print(f"Parquet-Dateien erstellt: {total_months}")
    print(f"Parquet-Dateien in: {base_parquet}")
    

if __name__ == "__main__":
    csv_path = "D:\Tankdaten\prices"
    parquet_path = "D:\Tankdaten\prices_parquet"
    convert_csvs_to_parquet(csv_path, parquet_path)
