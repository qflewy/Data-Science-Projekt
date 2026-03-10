from pathlib import Path
import polars as pl
import plotly.express as px
import pandas as pd
import geopandas as gpd
import osmnx as ox
from scipy import stats
from IPython.display import display


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

def mann_whitney_test_border_prices(median_price_path:Path,border_stations_file:Path,fuel_type:str):

    price_file = median_price_path / r'*/*.parquet'
    

    #initialize lazyframes & preprocess
    print("creating execution plan")
    border_lf = pl.scan_csv(border_stations_file).with_columns(pl.col("uuid").cast(pl.Utf8))

    price_lf = pl.scan_parquet(price_file).with_columns(pl.col("station_uuid").cast(pl.Utf8))

    preprocessed_lf = (price_lf.with_columns(pl.col("day").dt.year().alias("year"))
                       .join(border_lf,
                            left_on = "station_uuid",
                            right_on = "uuid",               
                            how = "inner"))
    
    # get df for yearly test
    yearly_lf = (preprocessed_lf.group_by(["year", "neighbour_country", "border_region", "station_uuid"])
                 .agg(pl.col(f"{fuel_type}_median").median().alias(f"{fuel_type}_median_price")))
    
    # get df for overall test
    overall_lf = (preprocessed_lf.group_by(["neighbour_country", "border_region", "station_uuid"])
                  .agg(pl.col(f"{fuel_type}_median").median().alias(f"{fuel_type}_median_price")))
    
    # collect data
    print("collecting data..")
    yearly_df = yearly_lf.collect()
    overall_df = overall_lf.collect()
    print(f"\n {yearly_df.head()}")
    print(f"\n {overall_df.head()}")

    print("data succesfully collected. continuing with the test.")

    # support function for the test
    def calculate_test(df:pl.DataFrame,country:str):
        #filter for conutry
        df_country = df.filter(pl.col("neighbour_country") == country)

        #get lists for border and surrounding region
        border_prices = (df_country.filter(pl.col("border_region") == "Border (0-15km)")
                        .get_column(f"{fuel_type}_median_price")
                        .cast(pl.Float64, strict = False)
                        .drop_nans()
                        .drop_nulls()
                        .to_list())
        surrounding_prices = (df_country.filter(pl.col("border_region") == "Surrounding (15-50km)")
                            .get_column(f"{fuel_type}_median_price")
                            .cast(pl.Float64, strict = False)
                            .drop_nans()
                            .drop_nulls()
                            .to_list())

        #check if we have enough stations for a test
        if len(border_prices) < 3 or len(surrounding_prices) < 3:
            return None
        
        #two sided mann whitney u test (-> checks for differences on both sides)
        stat, p_value = stats.mannwhitneyu(border_prices, surrounding_prices, alternative = "two-sided")

        #calculate median of both groups to see which one in cheaper
        median_border = pl.Series(border_prices).median()
        median_surrounding = pl.Series(surrounding_prices).median()

        return {
            "Country": country,
            "N_border": len(border_prices),
            "N_surrounding": len(surrounding_prices),
            "Median_border": round(median_border, 3),
            "Median_surrounding": round(median_surrounding, 3),
            "Price_difference": round(median_border - median_surrounding, 3), #negative means that border stations are cheaper
            "p_value": p_value,
            "Significant (5%)": True if p_value < 0.05 else False
         }
    
    #calculate the tests (yearly/overall) for each bordering country
    countries = yearly_df["neighbour_country"].unique().to_list()
    overall_results = []
    yearly_results = []

    #test over all the years
    for country in countries:
        result = calculate_test(overall_df, country)
        if result:
            overall_results.append(result)
    
    #test over each year separatly
    years = yearly_df["year"].unique().to_list()
    for country in countries:
        for year in years:
            filtered_df_yearly = yearly_df.filter(pl.col("year") == year)
            result = calculate_test(filtered_df_yearly, country)
            if result:
                result["year"] = year
                yearly_results.append(result)

    #show results
    overall_result_df = pd.DataFrame(overall_results)
    yearly_result_df = pd.DataFrame(yearly_results)
    
    yearly_result_df = yearly_result_df[["year", "Country", "Median_border", "Median_surrounding", 
                                         "Price_difference", "p_value", "Significant (5%)", "N_border", "N_surrounding"]]
    
    print("=== absolute results (over all years) ===")
    display(overall_result_df)

    print("\n=== yearly result (excerpt) ===")
    display(yearly_result_df.head(15))

def get_autobahn_stations(station_input_path:Path,autobahn_output_path:Path):

    stations_df = pd.read_csv(station_input_path)
    print("converting stations into geodata")
    stations_gdf = gpd.GeoDataFrame(
        stations_df,
        geometry = gpd.points_from_xy(stations_df["longitude"], stations_df["latitude"]),
        crs = "EPSG:4326"
    )

    stations_gdf = stations_gdf.to_crs(epsg = 32632) #convert coords to metric system

    #load german autobahnnetz from osm
    #filter for autobahn and service stations
    print("Loading autobahnnetz from open-street-maps")
    tags = {"highway": ["motorway", "service"]}
    autobahn = ox.features_from_place("Germany", tags = tags)
    autobahn = autobahn.to_crs(epsg = 32632)

    #create a 100m buffer around each autobahn to get close-by stations
    autobahn_buffered = autobahn.geometry.buffer(100)
    autobahn_polygon = autobahn_buffered.union_all()

    #filter stations that are on the autobahn
    print("checking which stations are on the autobahn")
    stations_gdf["is_autobahn"] = stations_gdf.geometry.intersects(autobahn_polygon)

    export_df = pd.DataFrame(stations_gdf.drop(columns=["geometry", "street", "house_number", "city", "name"]))

    export_df.to_csv(autobahn_output_path / r'autobahn_stations.csv', index = False)

    num_stations = export_df["is_autobahn"].sum()
    print(f"Success! we found {num_stations} stations on the autobahn.")
    print(f"File was saved as {autobahn_output_path}/autobahn_stations.csv")




if __name__=="__main__":
   
    stations_path = Path(r'/Users/sebastian/data-science-projekt/tankerkoenig_data/stations/stations.csv')
    border_output_path = Path(r'/Users/sebastian/data-science-projekt/tankerkoenig_data/stations')
    #get_borderregion_stations(stations_path, border_output_path)

    get_autobahn_stations(stations_path, border_output_path)