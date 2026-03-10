from pathlib import Path
import polars as pl
import plotly.express as px
import pandas as pd
import geopandas as gpd
import osmnx as ox

def show_median_price_heatmap_per_region(region_price_path:Path, region_path:Path, year:int, fuel_type:str):

    df_regions = pl.read_csv(region_path, dtypes={"leit_plz": pl.Utf8})
    regions = df_regions["leit_plz"].to_list()


    global_lf_list = []

    for region in regions:
        file_path = region_price_path / f"mean_median_price_region_{region}.csv"

        if file_path.exists():

            region_lf = pl.scan_csv(file_path)

            processed_region_lf = (
                region_lf.with_columns(pl.col("timestamp_utc").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f%z"))
                .filter(pl.col("timestamp_utc").dt.year() == year)
                .with_columns(pl.col("timestamp_utc").dt.strftime("%Y-%m").alias("month"),
                            pl.lit(region).alias("region")
                        )
                .group_by(["month", "region"])
                .agg(pl.col(f"{fuel_type}_median").median().alias(f"{fuel_type}_median_price"))
            )
            global_lf_list.append(processed_region_lf)
        else: continue

    global_df = pl.concat(global_lf_list).collect()

    display_df = (global_df.join(df_regions, left_on = "region", right_on = "leit_plz", how = "left").sort("month"))

    min_price = display_df[f"{fuel_type}_median_price"].min()
    max_price = display_df[f"{fuel_type}_median_price"].max()

    fig = px.scatter_map(
        display_df,
        lat = "avg_lat",
        lon = "avg_lng",
        color = f"{fuel_type}_median_price",
        center = {"lat": 51.16, "lon": 10.45},
        zoom = 4,
        map_style = "open-street-map",
        animation_frame = "month",
        hover_name = "region",
        title = f"Comparision of the median {fuel_type} price in year {year}",
        labels = {f"{fuel_type}_median_price": "Price (€)", "month": "Month"},
        range_color = [min_price, max_price],
        color_continuous_scale = "RdYlBu_r"
    )

    fig.update_layout(margin = {"r":0,"t":50,"l":0,"b":0})
    fig.update_traces(marker = dict(size = 15, opacity = 1))
    fig.show()

def get_borderregion_stations(station_input_path:Path, outputpath:Path):

    station_df = pd.read_csv(station_input_path, usecols=["uuid", "latitude", "longitude"])

    #convert coordinates into geodata to be able to calculate with distance in meter
    print("converting into geodata")
    station_geo_df = gpd.GeoDataFrame(station_df, 
                                      geometry = gpd.points_from_xy(station_df["longitude"], station_df["latitude"]),
                                      crs = "EPSG:4326")
    station_geo_df = station_geo_df.to_crs(epsg = 32632) #projection into metric system

    neighbor_countries = ["Denmark", "Poland", "Czechia", "Austria", "Switzerland", 
        "France", "Luxembourg", "Belgium", "Netherlands"]
    
    #loading the borders of neighbouring countries and calculating the distance from the stations
    dist_cloumns =[]

    for country in neighbor_countries:
        print(f"loading border for {country} and calculating distance.")

        #load country shape and convert coordinates to metrical system
        country_geo_df = ox.geocode_to_gdf(country)
        country_geo_df = country_geo_df.to_crs(epsg = 32632)
        print(country_geo_df.head())
        geom = country_geo_df.geometry.iloc[0]

        col_name = f"dist_{country}"
        station_geo_df[col_name] = station_geo_df.geometry.distance(geom) / 1000 #divide by 1000 to get kilometers
        dist_cloumns.append(col_name)
    
    #get closest border for each statino and extract country name and minimum distance
    station_geo_df["closest_border_col"] = station_geo_df[dist_cloumns].idxmin(axis=1)
    station_geo_df["neighbour_country"] = station_geo_df["closest_border_col"].str.replace("dist_", "")
    station_geo_df["dist_km"] = station_geo_df[dist_cloumns].min(axis = 1)

    #classify each station into either border, border surrounding region or far away (inland) (-> to avoid regional differences we ignore the latter)
    #the distances where arbitarily chosen, maybe change them later
    def get_zone(dist):
        if dist <= 15:
            return "Border (0-15km)"
        elif dist <= 50:
            return "Surrounding (15-50km)"
        else:
            return "Inland (>50km)"

    station_geo_df["border_region"] = station_geo_df["dist_km"].apply(get_zone)

    #filter sea border to denmark and bodensee
    mask_dk = (station_geo_df["neighbour_country"] == "Denmark") & ((station_geo_df["longitude"] > 9.6) | (station_geo_df["longitude"] < 8.8))
    mask_bodensee = (station_geo_df["neighbour_country"].isin(["Switzerland", "Austria"])) & \
        (station_geo_df["latitude"] < 47.8) & (station_geo_df["longitude"] > 8.9) & (station_geo_df["longitude"] < 9.8)
    station_geo_df.loc[mask_dk | mask_bodensee, "border_region"] = "sea border (ignore)"


    #filter to only keep border and surrounding stations & cleanup to save to csv
    filter_stations_gdf = station_geo_df[station_geo_df["border_region"].isin(["Border (0-15km)", "Surrounding (15-50km)"])].copy()
    necessary_columns = ["uuid", "latitude", "longitude", "neighbour_country", "dist_km", "border_region"]
    export_df = pd.DataFrame(filter_stations_gdf[necessary_columns])

    output_file = outputpath / "border_stations.csv"
    export_df.to_csv(output_file, index = False)
    
    print(f"Done. File saved in {output_file}")
    print(f"number of found border and surrounding stations:: {len(export_df)}")

