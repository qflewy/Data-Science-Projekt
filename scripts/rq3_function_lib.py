from pathlib import Path
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import geopandas as gpd
import osmnx as ox
from scipy import stats
from scipy.spatial import KDTree
import numpy as np
from linearmodels.iv import AbsorbingLS


def show_median_price_heatmap_per_region(region_price_path:Path, region_path:Path, year:int, fuel_type:str,monthly_median_prices:Path=None,use_median_file:bool=False):
    '''
    This function plots a map that dynamically projects the monthly median prices of a selected fuel type over a selected year for all post code Leitregionen. 
    If the monthly medians file exists, set use_median_file to True and give the filepath as input for monthly_median_prices to avoid having to compute the values from scratch. 
    Returns nothing.
    i: Path region_price_path, Path region_path, int year, string fuel_type, Path monthly_median_prices, bool use_median_file
    o: None
    '''
    #for website: if the median price file exists, use it to avoid computational overhead.
    if use_median_file & monthly_median_prices.exists():
        display_df = pl.read_parquet(monthly_median_prices)
    else:

        df_regions = pl.read_csv(region_path, dtypes={"leit_plz": pl.Utf8})
        regions = df_regions["leit_plz"].to_list()

        #create an empty list for all the regional lazyframes.
        global_lf_list = []

        #loop over all regions to calculate their monthly median.
        #this part was written with the help of gemini.
        for region in regions:
            file_path = region_price_path / f"mean_median_price_region_{region}.csv"

            if file_path.exists():

                region_lf = pl.scan_csv(file_path)

                processed_region_lf = (
                    region_lf.with_columns(pl.col("timestamp_utc").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f%z"))
                    .filter(pl.col("timestamp_utc").dt.year() == year) #filter for selected year.
                    .with_columns(pl.col("timestamp_utc").dt.strftime("%Y-%m").alias("month"),
                                pl.lit(region).alias("region")
                            )
                    .group_by(["month", "region"])
                    .agg(pl.col(f"{fuel_type}_median").median().alias(f"{fuel_type}_median_price")) #calculate monthly median price.
                )
                global_lf_list.append(processed_region_lf)
            else: continue

        #collect all lazyframes and concatenate to a dataframe.
        global_df = pl.concat(global_lf_list).collect()

        #join prices with regions to get the coordinates for the respective median price.
        display_df = (global_df.join(df_regions, left_on = "region", right_on = "leit_plz", how = "left").sort("month"))

    

    #get min and max price for an constant colorscale over the year.
    min_price = display_df[f"{fuel_type}_median_price"].min()
    max_price = display_df[f"{fuel_type}_median_price"].max()

    #create scatterplot.
    fig = px.scatter_map(
        display_df,
        lat = "avg_lat",
        lon = "avg_lng",
        color = f"{fuel_type}_median_price",
        center = {"lat": 51.16, "lon": 10.45}, #geographical centre of Germany.
        animation_frame = "month",
        hover_name = "region",
        title = f"Comparision of the median {fuel_type} price in year {year}",
        labels = {f"{fuel_type}_median_price": "Price (€)", "month": "Month"},
        range_color = [min_price, max_price],
        color_continuous_scale = "RdYlBu_r"
    )

    fig.update_layout(margin = {"r":0,"t":0,"l":0,"b":0},
                      map=dict(
                        style="open-street-map",
                        center=dict(lat=51.1657, lon=10.4515), #geographical center of germany.
                        zoom=6,
                        bounds=dict(
                            west=4.5,
                            east=16.5,
                            south=47.0,
                            north=55.2
                        )),
                        width=1200,
                        height=1400
    )
    fig.update_traces(marker = dict(size = 15, opacity = 1))

    fig.show()

#NOTE: parts of this function were written with the help of gemini.
def get_borderregion_stations(station_input_path:Path, outputpath:Path):
    '''
    This function calculates for each stations the distance to Germanys borders and sorts them into border, surrounding and inland groups. 
    Creates a Dataframe with the border and surrounding stations and their closest neighbouring country.
    This function uses the open-street-maps osmx api for the border coordinates.
    Safes the dataframe as csv to the selected output. Returns nothing.
    i: Path station_input_path, Path outputpath
    o: None
    '''

    station_df = pd.read_csv(station_input_path, usecols=["uuid", "latitude", "longitude", "brand", "post_code"])

    #convert coordinates into geodata to be able to calculate with distance in meter.
    print("converting into geodata")
    station_geo_df = gpd.GeoDataFrame(station_df, 
                                      geometry = gpd.points_from_xy(station_df["longitude"], station_df["latitude"]),
                                      crs = "EPSG:4326")
    station_geo_df = station_geo_df.to_crs(epsg = 32632) #projection into metric system.

    neighbor_countries = ["Denmark", "Poland", "Czechia", "Austria", "Switzerland", 
        "France", "Luxembourg", "Belgium", "Netherlands"]
    
    #loading the borders of neighbouring countries and calculating the distance from the stations.
    dist_cloumns =[]

    for country in neighbor_countries:
        print(f"loading border for {country} and calculating distance.")

        #load country shape and convert coordinates to metrical system.
        country_geo_df = ox.geocode_to_gdf(country)
        country_geo_df = country_geo_df.to_crs(epsg = 32632)
        print(country_geo_df.head())
        geom = country_geo_df.geometry.iloc[0]

        col_name = f"dist_{country}"
        station_geo_df[col_name] = station_geo_df.geometry.distance(geom) / 1000 #divide by 1000 to get kilometers.
        dist_cloumns.append(col_name)
    
    #get closest border for each statino and extract country name and minimum distance.
    station_geo_df["closest_border_col"] = station_geo_df[dist_cloumns].idxmin(axis=1)
    station_geo_df["neighbour_country"] = station_geo_df["closest_border_col"].str.replace("dist_", "")
    station_geo_df["dist_km"] = station_geo_df[dist_cloumns].min(axis = 1)

    #classify each station into either border, border surrounding region or far away (inland) (-> to avoid regional differences we ignore the latter).
    #the distances were arbitarily chosen, maybe change them later.
    def get_zone(dist):
        if dist <= 8:
            return "Border (0-8km)"
        elif dist <= 25:
            return "Surrounding (8-25km)"
        else:
            return "Inland (>25km)"

    station_geo_df["border_region"] = station_geo_df["dist_km"].apply(get_zone)

    #filter sea border to Denmark and Bodensee.
    mask_dk = (station_geo_df["neighbour_country"] == "Denmark") & ((station_geo_df["longitude"] > 9.6) | (station_geo_df["longitude"] < 8.8))
    mask_bodensee = (station_geo_df["neighbour_country"].isin(["Switzerland", "Austria"])) & \
        (station_geo_df["latitude"] < 47.8) & (station_geo_df["longitude"] > 8.9) & (station_geo_df["longitude"] < 9.8)
    station_geo_df.loc[mask_dk | mask_bodensee, "border_region"] = "sea border (ignore)"


    #filter to only keep border and surrounding stations & cleanup to save to csv.
    filter_stations_gdf = station_geo_df[station_geo_df["border_region"].isin(["Border (0-8km)", "Surrounding (8-25km)"])].copy()
    necessary_columns = ["uuid", "latitude", "longitude", "neighbour_country", "dist_km", "border_region", "brand"]
    export_df = pd.DataFrame(filter_stations_gdf[necessary_columns])

    output_file = outputpath / "lower_border_stations.csv"
    export_df.to_csv(output_file, index = False)
    
    print(f"Done. File saved in {output_file}")
    print(f"number of found border and surrounding stations:: {len(export_df)}")

#NOTE: parts of this function were written with the help of chatgpt.
def mann_whitney_test_border_prices(median_price_path:Path,border_stations_file:Path,fuel_type:str):
    '''
    Performs a Mann-Whitney-U-Test that tests whether two independent samples (border stations and non-border stations) have the same distribution.
    Returns two pandas DataFrames, one for the overall results and one for the separate yearly results.
    i: Path median_price_path, Path border_stations_file, string fuel_type
    o: pd.DataFrame overall_result_df, pd.DataFrame yearly_result_df
    '''

    price_file = median_price_path / r'*/*.parquet'
    

    #initialize lazyframes & preprocess.
    
    border_lf = pl.scan_csv(border_stations_file).with_columns(pl.col("uuid").cast(pl.Utf8))

    price_lf = pl.scan_parquet(price_file).with_columns(pl.col("station_uuid").cast(pl.Utf8))

    preprocessed_lf = (price_lf.with_columns(pl.col("day").dt.year().alias("year"))
                       .join(border_lf,
                            left_on = "station_uuid",
                            right_on = "uuid",               
                            how = "inner"))
    
    # get df for yearly test.
    yearly_lf = (preprocessed_lf.group_by(["year", "neighbour_country", "border_region", "station_uuid"])
                 .agg(pl.col(f"{fuel_type}_median").median().alias(f"{fuel_type}_median_price")))
    
    # get df for overall test.
    overall_lf = (preprocessed_lf.group_by(["neighbour_country", "border_region", "station_uuid"])
                  .agg(pl.col(f"{fuel_type}_median").median().alias(f"{fuel_type}_median_price")))
    
    #collect the data.
    yearly_df = yearly_lf.collect()
    overall_df = overall_lf.collect() 

    # support function for the test.
    def __calculate_test(df:pl.DataFrame,country:str):
        #filter for conutry.
        df_country = df.filter(pl.col("neighbour_country") == country)

        #get lists for border and surrounding region.
        border_prices = (df_country.filter(pl.col("border_region") == "Border (0-8km)")
                        .get_column(f"{fuel_type}_median_price")
                        .cast(pl.Float64, strict = False)
                        .drop_nans()
                        .drop_nulls()
                        .to_list())
        surrounding_prices = (df_country.filter(pl.col("border_region") == "Surrounding (8-25km)")
                            .get_column(f"{fuel_type}_median_price")
                            .cast(pl.Float64, strict = False)
                            .drop_nans()
                            .drop_nulls()
                            .to_list())

        #check if we have enough stations for a test.
        if len(border_prices) < 5 or len(surrounding_prices) < 5:
            return None
        
        #two sided mann whitney u test (-> checks for differences on both sides).
        _, p_value = stats.mannwhitneyu(border_prices, surrounding_prices, alternative = "two-sided")

        #calculate median of both groups to see which one in cheaper.
        median_border = pl.Series(border_prices).median()
        median_surrounding = pl.Series(surrounding_prices).median()

        return {
            "Country": country,
            "N_border": len(border_prices),
            "N_surrounding": len(surrounding_prices),
            "Median_border": round(median_border, 3),
            "Median_surrounding": round(median_surrounding, 3),
            "Price_difference": round(median_border - median_surrounding, 3), #negative means that border stations are cheaper.
            "p_value": p_value,
            "Significant (5%)": True if p_value < 0.05 else False
         }
    
    #calculate the tests (yearly/overall) for each bordering country.
    countries = yearly_df["neighbour_country"].unique().to_list()
    overall_results = []
    yearly_results = []

    #test over all the years.
    for country in countries:
        result = __calculate_test(overall_df, country)
        if result:
            overall_results.append(result)
    
    #test over each year separatly.
    years = yearly_df["year"].unique().to_list()
    for country in countries:
        for year in years:
            filtered_df_yearly = yearly_df.filter(pl.col("year") == year)
            result = __calculate_test(filtered_df_yearly, country)
            if result:
                result["year"] = year
                yearly_results.append(result)

    #safe and return the results.
    overall_result_df = pd.DataFrame(overall_results)
    yearly_result_df = pd.DataFrame(yearly_results)
    
    yearly_result_df = yearly_result_df[["year", "Country", "Median_border", "Median_surrounding", 
                                         "Price_difference", "p_value", "Significant (5%)", "N_border", "N_surrounding"]]
    
    return overall_result_df, yearly_result_df

def get_autobahn_stations(station_input_path:Path,autobahn_output_path:Path):
    '''
    This functions filters manually for buzzwords that indicate whether a gas station is on the Autobahn. It saves the Autobahn stations dataframe as a csv to the given output path. 
    Returns nothing.
    i: Path station_input_path, Path autobahn_output_path
    o: None
    '''

    stations_df = pd.read_csv(station_input_path)
    
    #filter the stations for buzzwords with regex.
    buzzword_pattern = r'(?i)(?:bab|raststätte|autobahn|rastanlage|rasthof|\bA\s?\d{1,3}\b)'

    # create masks for name, street & house number (because for some entries, the autobahn was in the house_number column).
    #this part was written with the help of gemini.
    mask_name = stations_df["name"].str.contains(buzzword_pattern, regex = True, na = False)
    mask_street = stations_df["street"].str.contains(buzzword_pattern, regex = True, na = False)
    mask_house_number = stations_df["house_number"].str.contains(buzzword_pattern, regex = True, na = False)

    # combine for all, if buzzword is part of either of these columns.
    stations_df["is_autobahn"] = mask_house_number | mask_name | mask_street
    
    #filter out non autobahn stations and export.
    stations_df = stations_df[stations_df["is_autobahn"] == True]
    export_df = stations_df[["uuid", "longitude", "latitude", "brand", "post_code"]]
    export_df.to_csv(autobahn_output_path / r'autobahn_stations.csv', index = False)

    num_stations = stations_df["is_autobahn"].sum()
    print(f"Success! we found {num_stations} stations on the autobahn.")
    print(f"File was saved as {autobahn_output_path}/autobahn_stations.csv")

def filter_autobahn_from_borders(border_file:Path,autobahn_file:Path,output_path:Path):
    '''
    This method removes all Autobahn stations from the border region stations. The result is saved as a new dataframe as csv to the given outputpath. Returns nothing.
    i: Path border_file, Path autobahn_file, Path output_path
    o: None
    '''
    autobahn_df = pl.read_csv(autobahn_file)
    border_df = pl.read_csv(border_file, schema_overrides={"post_code": pl.Utf8})

    #with the anti-join, all stations that are in the autobahn stations df are getting removed from the border stations df.
    no_autobahn_border_df = (border_df.join(
        autobahn_df,
        on = "uuid",
        how = "anti"
    ))

    no_autobahn_border_df.write_csv(output_path / "lower_non_autobahn_border_stations.csv") # (named "lower" because the threshold for border distance was lowered in the course of the project.)

def show_border_price_difference(median_price_path:Path,border_stations_file:Path,fuel_type:str,country:str,median_distributions_file:Path=None,use_distributions_file:bool=False):
    '''
    This function plots a boxplot with the distributions of the median fuel prices of a selected fuel type over time for a selected country. 
    For each year a box for the border region stations and a box for the surrounding region stations is plottet.
    For the website use: if the median distribution file exists, you can set use_distributions_file to True and give the Path as parameter. Returns nothing.
    i: Path median_price_path, Path border_stations_file, string fuel_type, string country, Path median_distribution_file, bool use_distributions_file
    o: None
    '''

    if use_distributions_file & median_distributions_file.exists():
        yearly_df = pl.read_parquet(median_distributions_file)
    else:

        price_file = median_price_path / r'*/*.parquet'
        
        #initialize lazyframes & preprocess.
        border_lf = pl.scan_csv(border_stations_file).with_columns(pl.col("uuid").cast(pl.Utf8))

        price_lf = pl.scan_parquet(price_file).with_columns(pl.col("station_uuid").cast(pl.Utf8))

        preprocessed_lf = (price_lf.with_columns(pl.col("day").dt.year().alias("year"))
                        .join(border_lf,
                                left_on = "station_uuid",
                                right_on = "uuid",               
                                how = "inner"))
        
        # get dataframe for yearly test.
        yearly_df = (preprocessed_lf.group_by(["year", "neighbour_country", "border_region", "station_uuid"])
                    .agg(pl.col(f"{fuel_type}_median").median().alias(f"{fuel_type}_median_price"))).collect()

    #filter and collect data for country.
    plot_df = (yearly_df.filter(
        (pl.col("neighbour_country") == country)
        ).select(["border_region", f"{fuel_type}_median_price", "year"])
        .to_pandas())
    plot_df["year"] = pd.to_datetime(plot_df["year"].astype(int).astype(str), format="%Y")
    #check if data is available.
    if plot_df.empty:
        print(f"no data found for {country}!")
        return
    


    fig = go.Figure()

    #the following part was written with the help of chatgpt.
    #create subplot for border and surrounding regions.
    for label, color in [("Border (0-8km)", "lightblue"), ("Surrounding (8-25km)", "violet")]:
        df = plot_df[plot_df["border_region"] == label]

        fig.add_trace(go.Box(
            x = plot_df["year"],
            y = df[f"{fuel_type}_median_price"],
            name = label,
            marker_color = color
        ))
    
    fig.update_layout(
        title = f"Yearly distribution of median {fuel_type} prices: border vs. surrounding-region stations ({country})",
        xaxis_title = "Year",
        yaxis_title = "Median price (€/liter)",
        boxmode = "group"
    )
    fig.update_xaxes(tickformat="%Y", dtick="M12")
    
    fig.show()
    
#NOTE: parts of this functions were written with the help of chatgpt.
def perform_matched_panel_regression_autobahn_stations(median_price_path:Path,stations_file:Path,autobahn_file:Path,fuel_type:str,statistic:str="mean",return_residuals:bool=False):
    '''
    This method performs a panel regression that estimates the coefficient of the factor 'autobahn' on the price of a selected fuel type. 
    It filters out the fixed effects of time and geographic proximity.
    For a robustness check, it's also possible to select the price median as the statistic. If needed, the function also returns the residual errors.
    i: Path median_price_path, Path stations_file, Path autobahn_file, string fuel_type, string statistic, bool return_residuals
    o: pd.DataFrame summary_df, pd.DataFrame analysis_panel, (pd.DataFrame residuals_df)
    '''

    #define fixed parameters (for regional clustering, ...).
    K_MATCHES = 5
    MAX_DISTANCE = 50
    
    # create stations and autobahn lf and merge them into a df, also classify the brand.
    stations_lf = (pl.scan_csv(stations_file).select(pl.col(["uuid", "brand", "latitude", "longitude"])))
    autobahn_lf = (pl.scan_csv(autobahn_file).select(pl.col(["uuid"])).with_columns(pl.lit(1).alias("autobahn")))

    stations_panel_df = (stations_lf
                .join(autobahn_lf, on = "uuid", how = "left")
                .with_columns([pl.col("autobahn").fill_null(0).cast(pl.Int8),
                               pl.col("brand").map_elements(__classify_brand, return_dtype=pl.Utf8).alias("brand_category")])
                .filter(pl.col("longitude").is_not_null() & pl.col("latitude").is_not_null())
                .collect())
    
    # create a price df.
    price_files = median_price_path / r'*/*.parquet'
    price_panel_df = (pl.scan_parquet(price_files)
                        .select([pl.col("station_uuid").alias("uuid"),
                               pl.col("day").cast(pl.Utf8).str.to_date(strict = False).alias("date"),
                               pl.col(f"{fuel_type}_{statistic}").cast(pl.Float64)])
                        .with_columns(pl.col("date").dt.year().alias("year"))
                        .filter(pl.col("uuid").is_not_null() & pl.col("date").is_not_null())
                        .collect())
    
    #build the panel for the analysis (combine the price and the stations dfs with the match map helper method).
    match_map = __build_match_map(stations_panel_df, K_MATCHES, MAX_DISTANCE)
    analysis_panel = (match_map.join(stations_panel_df
                                    .select(["uuid", "brand", "brand_category", "latitude", "longitude"]),
                                    left_on = "station_uuid",
                                    right_on = "uuid",
                                    how = "left")
                                .join(
                                    price_panel_df,
                                    left_on = "station_uuid",
                                    right_on = "uuid",
                                    how = "inner")
                                .filter(pl.col("date").is_not_null()))
    
    # perform the panel regression.
    outcome_col = f"{fuel_type}_{statistic}"

    panel_regression_df = (analysis_panel.select(["station_uuid","match_set_uuid", "date", "autobahn", "brand_category", pl.col(outcome_col).alias("y")])
                                         .filter(pl.col("y").is_not_null() & (pl.col("y") > 0))
                                         .to_pandas())
    panel_regression_df["date"] = pd.to_datetime(panel_regression_df["date"])
    panel_regression_df["year"] = panel_regression_df["date"].dt.year

    #iterate over the years.
    summary_rows = []
    residual_dfs = []

    for year in sorted(panel_regression_df["year"].dropna().unique()):
        year_df = panel_regression_df[panel_regression_df["year"] == year].copy()

        #define exogene variables (-> also get dummies to the brand categories).
        exog = pd.DataFrame({"autobahn": year_df["autobahn"].astype(float)})
        brand_dummies = pd.get_dummies(year_df["brand_category"], prefix = "brand", drop_first = True, dtype = float)
        exog = pd.concat([exog, brand_dummies], axis = 1)

        #define fixed effect that the modell needs to consider (regional and time fixed).
        absorb = pd.DataFrame({"match_set_uuid": pd.Categorical(year_df["match_set_uuid"]), "date": pd.Categorical(year_df["date"])})

        #define clustered standard error (treat the errors from one station over time as one cluster -> these errors are not independent from each other).
        clusters = pd.Categorical(year_df["station_uuid"]).codes.reshape(-1,1)

        #define the model (we use abosrbingls).
        model = AbsorbingLS(dependent = year_df["y"].astype(float),
                            exog = exog,
                            absorb = absorb,
                            drop_absorbed = True)
        result = model.fit(cov_type = "clustered",
                        clusters = clusters,
                        debiased = True)
        
        #calc a confidence intervall for each result.
        ci = result.conf_int()
        summary_rows.append({
                "year": year,
                "fuel_type": fuel_type,
                "statistic": statistic,
                "autobahn_coef": float(result.params["autobahn"]),
                "standard_error": float(result.std_errors["autobahn"]),
                "p_value": float(result.pvalues["autobahn"]),
                "ci_low": float(ci.loc["autobahn", "lower"]),
                "ci_high": float(ci.loc["autobahn", "upper"]),
                "n_observed": int(result.nobs)
            })
        
        #safe residual error dfs for wilcoxon test later on.
        if return_residuals:
            year_df["residuals"] = np.asarray(result.resids).reshape(-1)
            residual_dfs.append(year_df[["station_uuid", "match_set_uuid", "date", "year", "autobahn", "residuals"]])
    
    summary_df = pd.DataFrame(summary_rows)

    if return_residuals:
        residuals_df = pd.concat(residual_dfs, ignore_index = True)
        return summary_df,analysis_panel, residuals_df
    
    return summary_df, analysis_panel

def plot_yearly_autobahn_premium_line(yearly_df:pd.DataFrame,statistic:str="mean"):
    '''
    This method plots a line of the autobahn premium for a selected fuel type with surrounding confidence intervall over the years. Returns nothing.
    i: pd.DataFrame yearly_df, string statistic
    o: None
    '''

    #filter for selected statistic and format year column.
    yearly_df = yearly_df[yearly_df["statistic"] == statistic]
    yearly_df["year"] = pd.to_datetime(yearly_df["year"].astype(int).astype(str), format="%Y")

    fig = go.Figure()

    #define fuel types and color maps for each fuel type.
    fuel_types = ["diesel", "e5", "e10"]

    fill_color_map = {
        "diesel": "rgba(0,176,246,0.2)",
        "e5": "rgba(0,100,80,0.2)",
        "e10": "rgba(231,107,243,0.2)"
    }
    line_color_map = {
        "diesel": "rgba(255,255,255,0)",
        "e5": "rgba(255,255,255,0)",
        "e10": "rgba(255,255,255,0)"

    }
    color_map = {
        "diesel": "rgb(0,176,246)",
        "e5": "rgb(0,100,80)",
        "e10": "rgb(231,107,243)"
    }

    #iterate over the fuel types.
    for i, fuel in enumerate(fuel_types):
        df_sub = yearly_df[yearly_df["fuel_type"] == fuel].sort_values("year")

        #format the x and y values into desired format.
        #this part was written with the help of chatgpt.
        x_vals = df_sub["year"].to_list()
        x_rev = x_vals[::-1]

        y_upper = df_sub["ci_high"].to_list()
        y_lower = df_sub["ci_low"].to_list()[::-1]

        y_line = df_sub["autobahn_coef"].to_list()

        #create custom data for the hoverinfo.
        custom_data = list(zip(
            df_sub["year"].dt.year,
            df_sub["autobahn_coef"],
            df_sub["ci_high"],
            df_sub["ci_low"]
        ))
 
        #plot confidence intervalls.
        fig.add_trace(go.Scatter(
            x = x_vals + x_rev,
            y = y_upper + y_lower,
            fill = "toself",
            fillcolor = fill_color_map[fuel],
            line_color = line_color_map[fuel],
            name = fuel,
            showlegend = False,
            visible = (i == 0),
            hoverinfo = "skip"
            ))
        
        #plot autobahn premium line.
        fig.add_trace(go.Scatter(
            x = x_vals,
            y = y_line,
            mode = "lines+markers",
            line_color = color_map[fuel],
            showlegend = False,
            name = fuel,
            visible = (i == 0),
            customdata = custom_data,
            hovertemplate = (
                "Year: %{customdata[0]}<br>"
                "Autobahn-Premium: %{customdata[1]:.3f} €/liter<br>"
                "Upper confidance bound: %{customdata[2]:.3f} €/liter<br>"
                "Lower confidance bound: %{customdata[3]:.3f} €/liter<br>"
                "<extra></extra>"
            ) 

        ))
        

    #create dropdown to choose with fuel type to display.
    buttons = []
    for i, fuel in enumerate(fuel_types):
        visible = [False] * (2 * len(fuel_types))
        visible[2 * i] = True #conf. interval
        visible[2 * i +1] = True #line

        buttons.append(
            dict(
                label = fuel,
                method = "update",
                args = [{"visible": visible}, {"title": f"{fuel} price autobahnpremium with 95% confidence interval over the years"}]
            )
        )


    fig.update_layout(
        updatemenus = [
            dict(
                buttons = buttons,
                direction = "down",
                showactive = True,
                x = 1.05,
                y = 1
            )
        ],
        xaxis_title = "year",
        yaxis_title = "estimated autobahn premium (€/liter)",
        template = "plotly_white",
        title = f"{fuel_types[0]} price autobahnpremium with 95% confidence interval over the years"
        
    )
    fig.update_xaxes(tickformat="%Y", dtick="M12")
    
    fig.show()

def plot_autobahn_premium_barchart(yearly_df:pd.DataFrame,statistic:str="mean"):
    '''
    This function plots a bar chart where each bar represents one fuel type Autobahn premium in a year. So for each year, we have three separate bars. 
    The price statistic is "mean" by default, but can also be changed to "median" if wanted. Returns nothing.
    i: pd.DataFrame yearly_df, string statistic
    o: None 
    '''
    #filter & preprocessing.
    yearly_df = yearly_df[yearly_df["statistic"] == statistic]
    yearly_df["year"] = pd.to_datetime(yearly_df["year"].astype(int).astype(str), format="%Y")

    fig= go.Figure()

    #using the same colormap as in the lineplot.
    color_map = {
         "diesel": "rgb(0,176,246)",
        "e5": "rgb(0,100,80)",
        "e10": "rgb(231,107,243)"
    }

    #create a df with the values for each fuel type.
    y_diesel = yearly_df[yearly_df["fuel_type"] == "diesel"]
    y_e5 = yearly_df[yearly_df["fuel_type"] == "e5"]
    y_e10 = yearly_df[yearly_df["fuel_type"] == "e10"]


    #create separate bars for each fuel type.
    fig.add_trace(go.Bar(
        x = yearly_df["year"].dt.year,
        y = y_diesel["autobahn_coef"],
        name = "diesel",
        marker_color = color_map["diesel"],
        hovertemplate=(
                "Year: %{x}<br>"
                "Fuel: diesel<br>"
                "Autobahn premium: %{y:.3f} €/liter<br>"
                "<extra></extra>"
            )
    ))
    fig.add_trace(go.Bar(
        x = yearly_df["year"].dt.year,
        y = y_e5["autobahn_coef"],
        name = "e5",
        marker_color = color_map["e5"],
        hovertemplate=(
                "Year: %{x}<br>"
                "Fuel: e5<br>"
                "Autobahn premium: %{y:.3f} €/liter<br>"
                "<extra></extra>"
            )
    ))
    fig.add_trace(go.Bar(
        x = yearly_df["year"].dt.year,
        y = y_e10["autobahn_coef"],
        name = "e10",
        marker_color = color_map["e10"],
        hovertemplate=(
                "Year: %{x}<br>"
                "Fuel: e10<br>"
                "Autobahn premium: %{y:.3f} €/liter<br>"
                "<extra></extra>"
            )

    ))

    fig.update_layout(
        title = dict(text = "Development of the autobahn premium from 2014-2026"),
        xaxis_tickfont_size = 14,
        yaxis = dict(
            title = dict(
                text = "autobahn premium (€/liter)",
                font = dict(size = 16)
            )
        ),
        legend = dict(
            x = 0,
            y = 1.0,
            bgcolor = "rgba(255, 255, 255, 0)",
            bordercolor = "rgba(255, 255, 255, 0)"
        ),
        template = "plotly_white",
        barmode = "group",
        bargap = .15,
        bargroupgap = .1 
    )

    fig.show()


#NOTE: parts of this function were written with the help of chatgpt.
def perform_wilcoxon_variance_test_on_autobahn(residuals_df:pd.DataFrame,measure:str="mad",min_station_observations:int=30):
    '''
    This method perfoms a wilcoxon signed rank test on the residual errors of the fixed-effects panel regression for the autobahn premium. 
    The test compares the paired difference of the residual station volatility for Autobahn and each average control group for each year.
    The standard measure the test uses is median absolute deviation, it can also use the standard deviation as measure. 
    Also, the minimum required amount of observation per station, min_station_observations, can be changed.
    i: pd.DataFrame residuals_df, string measure, int min_station_observations
    o: pd.DataFrame results
    '''
    
    #check if meassure is valid.
    if measure not in ["mad", "sd"]:
        raise ValueError("measure has to be median absolute deviation (mad) or standard deviation (sd)")
    
    df = residuals_df.copy()

    #calculate the volatility per station and extract those with suffiecient sample size.
    station_volatility = (df.groupby(["match_set_uuid", "station_uuid", "year", "autobahn"], as_index = False)
                            .agg(n_obs = ("residuals", "size"),
                                 residuals_sd = ("residuals", "std"),
                                 residuals_mad = ("residuals", __mad)))
    
    station_volatility = station_volatility[station_volatility["n_obs"] >= min_station_observations].copy()

    volatility_col = "residuals_mad" if measure == "mad" else "residuals_sd"

    results = []

    #iterare over the years and compute wilcoxon test.
    for year in sorted(station_volatility["year"].dropna().unique()):
        yearly_df = station_volatility[station_volatility["year"] == year].copy()

        #create test and control group.
        test = (yearly_df[yearly_df["autobahn"] == 1]
                .rename(columns = {volatility_col: "test_volatility"})
                [["match_set_uuid", "test_volatility"]])
        
        controls = (yearly_df[yearly_df["autobahn"] == 0]
                    .groupby("match_set_uuid", as_index = False)
                    .agg(control_volatility = (volatility_col, "mean")))
        
        #create test-control pairs and check that they're not empty (-> if empty, skip for this year).
        paired = test.merge(controls, on = "match_set_uuid", how = "inner").dropna()
        if paired.empty:
            continue

        #calculate volatility difference and round to 12 decimals.
        diff = np.round((paired["test_volatility"] - paired["control_volatility"]).to_numpy(), 12)

        #if there if no difference, drop the pair (default zero handling for wilcoxon test).
        diff = diff[diff != 0]

        #return empty results if there are no non zero differences.
        if len(diff) == 0:
            results.append({
                "year": year,
                "measure": measure,
                "n_pairs": 0,
                "mean_test_volatility": np.nan,
                "mean_control_volatility": np.nan,
                "median_difference": 0.0,
                "wilcoxon_stat": np.nan,
                "p_value": np.nan
            })
            continue

        #calculate wilcoxon results.
        res = stats.wilcoxon(diff, alternative = "two-sided", method = "auto")

        results.append({
                "year": year,
                "measure": measure,
                "n_pairs": len(diff),
                "mean_test_volatility": paired["test_volatility"].mean(),
                "mean_control_volatility": paired["control_volatility"].mean(),
                "median_difference": np.median(diff),
                "wilcoxon_stat": res.statistic,
                "p_value": res.pvalue
            })
    
    return pd.DataFrame(results)

def plot_wilcoxon_results_loolipop(wilcoxon_df:pd.DataFrame):
    '''
    Plots a lollipop plot for the median volatility difference of autobahn station prices and non autobahn station prices. 
    The difference is positive, when the autobahn volatility is higher than the non autobahn volatility. Returns nothing.
    i: pd.Dataframe wilcoxon_df
    o: None
    '''

    # Build one single trace for all stems using None separators.
    # this part was written with the help of chatgpt.
    x_stems = []
    y_stems = []
    wilcoxon_df["year"] = pd.to_datetime(wilcoxon_df["year"].astype(int).astype(str), format="%Y")

    for _, row in wilcoxon_df.iterrows():
        x_stems.extend([row["year"], row["year"], None])
        y_stems.extend([0, row["median_difference"], None])

    fig = go.Figure()

    # create the vertical lines to the points.
    fig.add_trace(
        go.Scatter(
            x=x_stems,
            y=y_stems,
            mode="lines",
            line=dict(width=2),
            showlegend=False,
            hoverinfo="skip"
        )
    )

    custom_data = list(zip(
        wilcoxon_df["year"].dt.year,
        wilcoxon_df["median_difference"]
    ))

    # create the lollipop heads.
    fig.add_trace(
        go.Scatter(
            x=wilcoxon_df["year"],
            y=wilcoxon_df["median_difference"],
            mode="markers",
            marker=dict(size=12),
            name="median volatility difference",
            customdata=custom_data,
            hovertemplate=(
                "Year: %{customdata[0]}<br>"
                "Median volatility difference: %{customdata[1]:.4f} €/liter<br>"
                "<extra></extra>"
            )

        )
    )

    fig.update_layout(
        title="Autobahn vs non autobahn stations median volatility difference over the years",
        xaxis_title="year",
        yaxis_title="median difference (€/liter)",
        template="plotly_white"
    )
    fig.update_xaxes(tickformat="%Y", dtick="M12")

    fig.show()


#---- helper methods ----
    
#classify each brand into one of these categoties: brand, non brand, not defined/unknown.
#NOTE: this method was written with the help of chatgpt.
def __classify_brand(brand:str):
    '''
    This method classifies the brand of a gas station using its name and a predefined list of buzzwords that indicate the brand class of a station. 
    The brand classes are "unbranded", "branded" and "not defined/unknown".
    i: string brand
    o: string brand_class
    '''
    #define what to consider as brand station and what as free
    BRANDS = ["ARAL", "SHELL", "JET", "TOTAL", "TOTAL ENERGIES", "ESSO", "AVIA", "AVIA EXPRESS"
    "HEM", "HOYER", "ORLEN", "Q1", "STAR", "RAIFFEISEN", "AGIP",
    "ENI", "OMV", "OIL!", "WESTFALEN"]

    NON_BRAND_KEYWORDS = ["FREIE", "MARKENFREI", "OHNE MARKE",
    "UNBRANDED", "NONAME", "NO NAME", "FREIE TANKSTELLE"]

    if brand is None:
        return "unbranded"
    brand = str(brand).strip().upper()
    brand = " ".join(brand.replace("-", " ").replace("_", " ").split())

    if any(keyword in brand for keyword in BRANDS):
        return "branded"
    if brand == "" or brand in ["NULL", "NONE", "NAN", "UNBEKANNT"]:
        return "unbranded"
    if any(keyword in brand for keyword in NON_BRAND_KEYWORDS):
        return "unbranded"
        
    return "not defined/unknown"
    
# we match each autobahn station with a k-dim tree to its nearest non autobahn neighbours to get regional proximity for the controll groups.
# By doing this, we mitigate the risk of other geographical effects. For calculation, we use geopandas to convert the coordinates to a 2d metric system format.
# written wirth the help of chatgpt
def __build_match_map(stations:pl.DataFrame, k:int, max_dist:int):
    pdf = stations.select(["uuid", "autobahn", "longitude", "latitude"]).to_pandas()

    gdf = gpd.GeoDataFrame(pdf,
                            geometry = gpd.points_from_xy(pdf["longitude"], pdf["latitude"]),
                            crs = "EPSG:4326").to_crs(epsg = 32632)
        
    #split dataset.
    test = gdf[gdf["autobahn"] == 1].reset_index(drop = True)
    controll = gdf[gdf["autobahn"] == 0].reset_index(drop = True)

    if test.empty:
        raise ValueError("No autobahn stations found")
    if controll.empty:
        raise ValueError("No non autobahn stations found")
        
    # to avoid érrors, make sure k doesnt exceed number of available controll stations.
    k_eff = min(k, len(controll))

    #combine lat and lng into one column.
    test_xy = np.column_stack([test.geometry.x, test.geometry.y])
    controll_xy = np.column_stack([controll.geometry.x, controll.geometry.y])

    # build tree.
    tree = KDTree(controll_xy)
    distances, indices = tree.query(test_xy, k = k_eff)

    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
 
    rows = []

    #for each autobahn station, query the tree to get the closest non autobahn stations.
    for i in range(len(test)):
        test_id = test.loc[i, "uuid"]

        #add respective autobahn station to our list.
        rows.append({
                "match_set_uuid" : test_id,
                "station_uuid": test_id,
                "autobahn": 1,
                "dist_km": 0.0
            })

        for dist, idx in zip(distances[i], indices[i]):
            dist_km = float(dist) / 1000.0 #convert from meters to km.

            if max_dist is not None and dist_km > max_dist:
                continue

            rows.append({
                    "match_set_uuid": test_id,
                    "station_uuid": controll.loc[int(idx), "uuid"],
                    "autobahn": 0,
                    "dist_km": dist_km
                })
    match_map = pl.DataFrame(rows).unique(subset = ["match_set_uuid", "station_uuid"])

    #filter out invalid sets (see if there are enough controll points).
    valid_sets = (match_map
                      .group_by("match_set_uuid")
                      .agg([pl.len().alias("n"),
                            pl.col("autobahn").sum().alias("n_test")])
                      .filter(((pl.col("n") >= 2) & pl.col("n_test") == 1))
                      .select("match_set_uuid")                    
        )

    return match_map.join(valid_sets, on = "match_set_uuid", how = "inner")

# calculate median absolute devianion.
#written with the help of chatgpt.
def __mad(x):
    '''
    This method calculates the median absolute deviation for a given one-dim array-like/pd.Series.
    i: pd.Series / one-dim array-like x
    o: float mad
    '''
    x = np.asarray(x, dtype=float)
    median = np.median(x)
    return np.median(np.abs(x - median))

#saves plot to a png with sufficient resolution for poster.
#Partly written with help of chatgpt.
def save_png(fig, img_name:Path, legend:bool=False):
    '''
    This method saves plotly figures with high resolution (for the poster) to the given output path. 
    Before saving the plot, the method adjusts the text to an appropriate size. If the plot has legend, set legend to True so it also adjusts the legends font size before saving. 
    The chosen figure name should contain the suffix ".png". Returns nothing.
    i: plotly Figure fig, Path img_name, bool legend
    o: None
    '''

    px_w = 4200
    px_h = 2250

    fig = fig.full_figure_for_development(warn = False)
    fig.update_layout(
        autosize = False,
        width = px_w/2,
        height = px_h,
        font = dict(size=94),
        title = dict(font = dict(size =100),
                     y = .99,
                     x = .5,
                     xanchor = "center",
                     yanchor = "top")
    )
    if legend:
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                font = dict(size = 85),
                y= .97,
                xanchor="right",
                itemsizing="constant",
                x=0
            ),
            legend_title = None
        )

    fig.update_xaxes(tickfont = dict(size = 80), title_font = dict(size = 90))
    fig.update_yaxes(tickfont = dict(size = 80), title_font = dict(size = 90))

    img_path = Path(r'/Users/sebastian/data-science-projekt/rq_results') #change path to own pc
    fig.write_image(img_path/img_name,
                    width=px_w,
                    height=px_h,
                    scale=1)

if __name__=="__main__":
   
    stations_path = Path(r'/Users/sebastian/data-science-projekt/tankerkoenig_data/stations/stations.csv')
    border_output_path = Path(r'/Users/sebastian/data-science-projekt/tankerkoenig_data/stations')
    summany_file = Path(r'/Users/sebastian/data-science-projekt/rq_results/rq3_panel_summary.csv')
    #get_borderregion_stations(stations_path, border_output_path)
    #get_autobahn_stations(stations_path, border_output_path)
    #filter_autobahn_from_borders(border_output_path / "lower_border_stations.csv", border_output_path / "autobahn_stations.csv", border_output_path)
    
