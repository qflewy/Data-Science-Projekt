import polars as pl

from pathlib import Path

def create_station_regions(input_path:Path, output_path:Path):
    '''

    Clusters the stations by PLZ Leitregionen.

    NOTE: Parts of this code where written with the help of gemini.

    in: the paths for input and output
    out: separate sv files with the uuids per region.
    '''

    
    
    df = pl.read_csv(input_path, 
                     null_values=["nicht", "NA", "N/A", ""],
                     schema_overrides={"post_code": pl.String})
  

    processed_df = df.with_columns(pl.col("post_code").str.slice(0,2).alias("Leitregion"))

    regions_df = processed_df.partition_by("Leitregion")
    
    output_path.mkdir(parents=True, exist_ok=True)

    for region in regions_df:
        try:

            current_region = region["Leitregion"][0]

            file_path = output_path / f"uuids_region_{current_region}.csv"

            region.select("uuid").write_csv(file_path)
        except Exception as e:
            #ignore the missing post codes.
            pass  
                    




