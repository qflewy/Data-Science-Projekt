from pathlib import Path
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import geopandas as gpd
import osmnx as ox
from scipy import stats
from scipy.spatial import KDTree
from IPython.display import display
import numpy as np
from linearmodels.iv import AbsorbingLS

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

    station_df = pd.read_csv(station_input_path, usecols=["uuid", "latitude", "longitude", "brand", "post_code"])

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
        if dist <= 8:
            return "Border (0-8km)"
        elif dist <= 25:
            return "Surrounding (8-25km)"
        else:
            return "Inland (>25km)"

    station_geo_df["border_region"] = station_geo_df["dist_km"].apply(get_zone)

    #filter sea border to denmark and bodensee
    mask_dk = (station_geo_df["neighbour_country"] == "Denmark") & ((station_geo_df["longitude"] > 9.6) | (station_geo_df["longitude"] < 8.8))
    mask_bodensee = (station_geo_df["neighbour_country"].isin(["Switzerland", "Austria"])) & \
        (station_geo_df["latitude"] < 47.8) & (station_geo_df["longitude"] > 8.9) & (station_geo_df["longitude"] < 9.8)
    station_geo_df.loc[mask_dk | mask_bodensee, "border_region"] = "sea border (ignore)"


    #filter to only keep border and surrounding stations & cleanup to save to csv
    filter_stations_gdf = station_geo_df[station_geo_df["border_region"].isin(["Border (0-8km)", "Surrounding (8-25km)"])].copy()
    necessary_columns = ["uuid", "latitude", "longitude", "neighbour_country", "dist_km", "border_region", "brand"]
    export_df = pd.DataFrame(filter_stations_gdf[necessary_columns])

    output_file = outputpath / "lower_border_stations.csv"
    export_df.to_csv(output_file, index = False)
    
    print(f"Done. File saved in {output_file}")
    print(f"number of found border and surrounding stations:: {len(export_df)}")

def mann_whitney_test_border_prices(median_price_path:Path,border_stations_file:Path,fuel_type:str):

    price_file = median_price_path / r'*/*.parquet'
    

    #initialize lazyframes & preprocess
    
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
    
    yearly_df = yearly_lf.collect()
    overall_df = overall_lf.collect() 

    # support function for the test
    def calculate_test(df:pl.DataFrame,country:str):
        #filter for conutry
        df_country = df.filter(pl.col("neighbour_country") == country)

        #get lists for border and surrounding region
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

        #check if we have enough stations for a test
        if len(border_prices) < 5 or len(surrounding_prices) < 5:
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
    
    #filter the stations for buzzwords with regex
    buzzword_pattern = r'(?i)(?:bab|raststätte|autobahn|rastanlage|rasthof|\bA\s?\d{1,3}\b)'

    # create masks for name, street & house number (because for some entries, the autobahn was in the house_number column)
    mask_name = stations_df["name"].str.contains(buzzword_pattern, regex = True, na = False)
    mask_street = stations_df["street"].str.contains(buzzword_pattern, regex = True, na = False)
    mask_house_number = stations_df["house_number"].str.contains(buzzword_pattern, regex = True, na = False)

    # combine for all, if buzzword is part of either of these columns
    stations_df["is_autobahn"] = mask_house_number | mask_name | mask_street
    
    #filter out non autobahn stations and export
    stations_df = stations_df[stations_df["is_autobahn"] == True]
    export_df = stations_df[["uuid", "longitude", "latitude", "brand", "post_code"]]
    export_df.to_csv(autobahn_output_path / r'autobahn_stations.csv', index = False)

    num_stations = stations_df["is_autobahn"].sum()
    print(f"Success! we found {num_stations} stations on the autobahn.")
    print(f"File was saved as {autobahn_output_path}/autobahn_stations.csv")

def filter_autobahn_from_borders(border_file:Path,autobahn_file:Path,output_path:Path):
    autobahn_df = pl.read_csv(autobahn_file)
    border_df = pl.read_csv(border_file, schema_overrides={"post_code": pl.Utf8})

    no_autobahn_border_df = (border_df.join(
        autobahn_df,
        on = "uuid",
        how = "anti"
    ))

    no_autobahn_border_df.write_csv(output_path / "lower_non_autobahn_border_stations.csv")

def show_border_price_difference(median_price_path:Path,border_stations_file:Path,fuel_type:str,country:str,year:int):
    price_file = median_price_path / r'*/*.parquet'
    

    #initialize lazyframes & preprocess
    
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
    
    # collect data
    yearly_df = yearly_lf.collect()

    #filter data for country and year
    plot_df = (yearly_df.filter(
        (pl.col("neighbour_country") == country) & 
        (pl.col("year") == year) #&
        #(pl.col(f"{fuel_type}_median_price"))
        ).select(["border_region", f"{fuel_type}_median_price"])
        .to_pandas())
    #check if data is available
    if plot_df.empty:
        print(f"no data found for {country} in {year}!")
        return
    
    #create graph
    fig = px.histogram(
        plot_df,
        x = f"{fuel_type}_median_price",
        color = "border_region",
        barmode = "overlay",
        histnorm = "probability",
        marginal = "box",
        nbins = 40,
        color_discrete_map = {
            "Border (0-8km)": "red",
            "Surrounding (8-25km)": "blue"
        },
        title = f"{fuel_type} price distribution: Border vs. Surrounding region for {country} ({year})",
        labels = {f"{fuel_type}_median_price": "daily median (€)", "border_region": "zone"}
    )

    fig.update_traces(opacity = 0.7)
    fig.update_layout(template = "plotly_white",
                      xaxis_title = "Price (€)",
                      yaxis_title = "relative probality density",
                      legend = dict(
                          yanchor = "top",
                          y = .99,
                          xanchor = "right",
                          x = .99
                      )
                    )
    
    fig.show()
    
def perform_matched_panel_regression_autobahn_stations(median_price_path:Path,stations_file:Path,autobahn_file:Path,fuel_type:str,statistic:str="mean",return_residuals:bool=False):

    #define fixed parameters (for regional clustering, ...)
    K_MATCHES = 5
    MAX_DISTANCE = 50
    
    # create stations and autobahn lf and merge them into a df, also classify the brand
    stations_lf = (pl.scan_csv(stations_file).select(pl.col(["uuid", "brand", "latitude", "longitude"])))
    autobahn_lf = (pl.scan_csv(autobahn_file).select(pl.col(["uuid"])).with_columns(pl.lit(1).alias("autobahn")))

    stations_panel_df = (stations_lf
                .join(autobahn_lf, on = "uuid", how = "left")
                .with_columns([pl.col("autobahn").fill_null(0).cast(pl.Int8),
                               pl.col("brand").map_elements(__classify_brand, return_dtype=pl.Utf8).alias("brand_category")])
                .filter(pl.col("longitude").is_not_null() & pl.col("latitude").is_not_null())
                .collect())
    
    # create a price df
    price_files = median_price_path / r'*/*.parquet'
    price_panel_df = (pl.scan_parquet(price_files)
                        .select([pl.col("station_uuid").alias("uuid"),
                               pl.col("day").cast(pl.Utf8).str.to_date(strict = False).alias("date"),
                               pl.col(f"{fuel_type}_{statistic}").cast(pl.Float64)])
                        .with_columns(pl.col("date").dt.year().alias("year"))
                        .filter(pl.col("uuid").is_not_null() & pl.col("date").is_not_null())
                        .collect())
    
    #build the panel for the analysis (combine the price and the stations dfs with the match map helper method)
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
    
    # perform the panel regression 
    outcome_col = f"{fuel_type}_{statistic}"

    panel_regression_df = (analysis_panel.select(["station_uuid","match_set_uuid", "date", "autobahn", "brand_category", pl.col(outcome_col).alias("y")])
                                         .filter(pl.col("y").is_not_null() & (pl.col("y") > 0))
                                         .to_pandas())
    panel_regression_df["date"] = pd.to_datetime(panel_regression_df["date"])
    panel_regression_df["year"] = panel_regression_df["date"].dt.year

    #iterate over the years
    summary_rows = []
    residual_dfs = []

    for year in sorted(panel_regression_df["year"].dropna().unique()):
        year_df = panel_regression_df[panel_regression_df["year"] == year].copy()

        #define exogene variables (-> also get dummies to the brand categories)
        exog = pd.DataFrame({"autobahn": year_df["autobahn"].astype(float)})
        brand_dummies = pd.get_dummies(year_df["brand_category"], prefix = "brand", drop_first = True, dtype = float)
        exog = pd.concat([exog, brand_dummies], axis = 1)

        #define fixed effect that the modell needs to consider (regional and time fixed)
        absorb = pd.DataFrame({"match_set_uuid": pd.Categorical(year_df["match_set_uuid"]), "date": pd.Categorical(year_df["date"])})

        #define clustered standard error (treat the errors from one station over time as one cluster -> these errors are not independent from each other)
        clusters = pd.Categorical(year_df["station_uuid"]).codes.reshape(-1,1)

        #define the model (we use abosrbingls)
        model = AbsorbingLS(dependent = year_df["y"].astype(float),
                            exog = exog,
                            absorb = absorb,
                            drop_absorbed = True)
        result = model.fit(cov_type = "clustered",
                        clusters = clusters,
                        debiased = True)
        
        #calc a confidence intervall for each result
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
        
        #safe residual error dfs for wilcoxon test
        if return_residuals:
            year_df["residuals"] = np.asarray(result.resids).reshape(-1)
            residual_dfs.append(year_df[["station_uuid", "match_set_uuid", "date", "year", "autobahn", "residuals"]])
    
    summary_df = pd.DataFrame(summary_rows)

    if return_residuals:
        residuals_df = pd.concat(residual_dfs, ignore_index = True)
        return summary_df,analysis_panel, residuals_df
    
    return summary_df, analysis_panel

def plot_yearly_autobahn_premium_line(yearly_df:pd.DataFrame,statistic:str="mean"):

    yearly_df = yearly_df[yearly_df["statistic"] == statistic]

    fig = px.line(
        yearly_df,
        x = "fuel_type",
        y = "autobahn_coef",
        color = "fuel_type",
        markers = True,
        hover_data={
            "year": True,
            "fuel_type": True,
            "statistic": True,
            "autobahn_coef":":.3f",
            "ci_low":":.3f",
            "ci_high":":.3f"},
        title = "development of the autobahn premium over time"
    )

    #add confidence intervals for the fuel types
    for fuel in yearly_df["fuel_type"].unique():
        df = yearly_df[yearly_df["fuel_type"] == fuel].sort_values("year")
        
        fig.add_trace(go.Scatter(
            x = pd.concat([df["year"], df["year"][::-1]]),
            y = pd.concat([df["ci_high"], df["ci_low"][::-1]]),
            fill = "toself",
            mode = "none",
            line = dict(width=0),
            name = f"{fuel} CI",
            showlegend = False,
            opacity = .2,
            hoverinfo = "skip"
        ))
    fig.update_layout(
        xaxis_title = "year",
        yaxis_title = "estimated autobahn premium (€/liter)",
        template = "plotly_white"
    )
    fig.show()

def plot_autobahn_premium_histogram(yearly_df:pd.DataFrame):
    fig = px.histogram(
        yearly_df,
        x = "autobahn_coef",
        color = "fuel_type",
        barmode = "overlay",
        nbins = 15,
        title = "distribution of the yearly autobahn premium"
    )
    fig.update_layout(
        xaxis_title = "estimated autobahn premium (€/liter)",
        yaxis_title = "count",
        template = "plotly_white"
    )
    fig.show()

def plot_autobahn_premium_boxplot(yearly_df:pd.DataFrame):
    fig = px.box(
        yearly_df,
        x = "fuel_type",
        y = "autobahn_coef",
        color = "fuel_type",
        points = "all",
        title = "yearly autobahn premium by fuel type"
    )
    fig.update_layout(
        xaxis_title = "fuel type",
        yaxis_title = "estimated autobahn premium (€/liter)",
        template = "plotly_white",
        showlegend = False
    )
    fig.show()

def plot_autobahn_premium_barchart(yearly_df:pd.DataFrame):

    years = sorted(yearly_df["year"].unique())

    fig= go.Figure()

    color_map = {
        "diesel": "royalblue",
        "e5": "tomato",
        "e10": "mediumseagreen"
    }

    for i, year in enumerate(years):
        df_sub = yearly_df[yearly_df["year"] == year]

        fig.add_trace(go.Bar(
            x = df_sub["fuel_type"],
            y = df_sub["autobahn_coef"],
            name = str(year),
            visible = (i == 0),
            width = 0.45,
            marker_color = df_sub["fuel_type"].map(color_map)
        ))
    buttons  = []
    for i, year in enumerate(years):
        visible = [False] * len(years)
        visible[i] = True

        buttons.append(
            dict(
                label = str(year),
                method = "update",
                args = [{"visible": visible}, {"title": f"premium comparison for fuel types for {year}"}]
            )
        )
    fig.update_layout(
        title = f"premium comparison for fuel types for {years[0]}",
        updatemenus = [
            dict(
                buttons = buttons,
                direction = "down",
                showactive = True,
                x = 1.05,
                y = 1
        )],
        xaxis_title = "fuel types",
        yaxis_title = "autobahn premium (€/liter)",
        bargap = 0.35,
        hovermode = "closest"
    )
    fig.update_yaxes(showspikes = False)
    fig.update_xaxes(showspikes = False)

    fig.show()

#NOTE:funktioniert noch nicht wie gewollt (verscheidene symbole für autobahn oder nicht klappen nicht)
def plot_station_price_map(analyses_panel:pl.DataFrame,fuel_type:str,statistic="median"):
    
    price_col = f"{fuel_type}_{statistic}"

    # create a pandas df with the necessary data for a map plot
    plot_df = (analyses_panel.select(["station_uuid", "autobahn", "brand_category", "latitude", "longitude", pl.col(price_col).alias("price")])
                       .filter(pl.col("price").is_not_null() & pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null())
                       .group_by(["station_uuid", "autobahn", "brand_category", "latitude", "longitude"])
                       .agg(pl.median("price").alias("station_price_median"))
                       .sort("station_price_median")
                       .to_pandas())
    #create plot
    fig = go.Figure()

    symbol_map = {
        0: ("circle", "non-autobahn"),
        1: ("diamond", "autobahn")
    }

    cmin = plot_df["station_price_median"].min()
    cmax = plot_df["station_price_median"].max()

    for i, (cat, (symb, label)) in enumerate(symbol_map.items()):
        df_sub = plot_df[plot_df["autobahn"] == cat]

        fig.add_trace(go.Scattermap(
            lat=df_sub["latitude"],
            lon=df_sub["longitude"],
            mode="markers",
            name=label,
            marker=dict(
                size=9,
                symbol=symb,
                color=df_sub["station_price_median"],
                colorscale="Turbo",
                cmin=cmin,
                cmax=cmax,
                showscale=(i == 0),
                colorbar=dict(title="station price median") if i == 0 else None
            ),
            customdata=df_sub[["station_uuid", "autobahn", "brand_category", "station_price_median"]],
            hovertemplate=
                "UUID: %{customdata[0]}<br>"
                "Autobahn: %{customdata[1]}<br>"
                "Brand category: %{customdata[2]}<br>"
                "Median price: %{customdata[3]:.3f}<extra></extra>"
        ))

    fig.update_layout(
        title=f"{statistic} price map for {fuel_type} for each station",
        map=dict(
            style="open-street-map",
            zoom=4,
            center=dict(
                lat=plot_df["latitude"].mean(),
                lon=plot_df["longitude"].mean()
            )
        ),
        template="plotly_white",
        margin=dict(l=10, r=10, t=50, b=10)
    )

    fig.show()

def perform_wilcoxon_variance_test_on_autobahn(residuals_df:pd.DataFrame,measure:str="mad",min_station_observations:int=30):
    
    #check if meassure is valid
    if measure not in ["mad", "sd"]:
        raise ValueError("measure has to be mean absolute deviation (mad) or standard deviation (sd)")
    
    df = residuals_df.copy()

    #calculate the volatility per station and extract those with suffietient sample size
    station_volatility = (df.groupby(["match_set_uuid", "station_uuid", "year", "autobahn"], as_index = False)
                            .agg(n_obs = ("residuals", "size"),
                                 residuals_sd = ("residuals", "std"),
                                 residuals_mad = ("residuals", __mad)))
    
    station_volatility = station_volatility[station_volatility["n_obs"] >= min_station_observations].copy()

    volatility_col = "residuals_mad" if measure == "mad" else "residuals_sd"

    results = []

    #iterare over the years and compute wilcoxon test
    for year in sorted(station_volatility["year"].dropna().unique()):
        yearly_df = station_volatility[station_volatility["year"] == year].copy()

        #create test and control group
        test = (yearly_df[yearly_df["autobahn"] == 1]
                .rename(columns = {volatility_col: "test_volatility"})
                [["match_set_uuid", "test_volatility"]])
        
        controls = (yearly_df[yearly_df["autobahn"] == 0]
                    .groupby("match_set_uuid", as_index = False)
                    .agg(control_volatility = (volatility_col, "mean")))
        
        #create test-control pairs and check if theyre not empty (-> if empty, skip for this year)
        paired = test.merge(controls, on = "match_set_uuid", how = "inner").dropna()
        if paired.empty:
            continue

        #calculate volatility difference
        diff = np.round((paired["test_volatility"] - paired["control_volatility"]).to_numpy(), 12)

        #if there if no difference, drop the pair ( default zero handling for wilcoxon test)
        diff = diff[diff != 0]

        #return empty results if there are no non zero differences
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

        #calculate wilcoxon results
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


#---- helper methods ----
    
#classify each brand into one of these categoties: brand, non brand, not defined/unknown
#written with the help of chatgpt
def __classify_brand(brand:str):
    #define fixed variables
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
    
# we match each autobahn station with a k-dim tree to its nearest non autobahn neighbours to get regional proximity for the controll groups
# By doing this, we mitigate the risk of other geographical effects. For calculation, we use geopandas to convert the coordinates to a 2d metric system format.
#written wirth the help of chatgpt
def __build_match_map(stations:pl.DataFrame, k:int, max_dist:int):
    pdf = stations.select(["uuid", "autobahn", "longitude", "latitude"]).to_pandas()

    gdf = gpd.GeoDataFrame(pdf,
                            geometry = gpd.points_from_xy(pdf["longitude"], pdf["latitude"]),
                            crs = "EPSG:4326").to_crs(epsg = 32632)
        
    #split dataset
    test = gdf[gdf["autobahn"] == 1].reset_index(drop = True)
    controll = gdf[gdf["autobahn"] == 0].reset_index(drop = True)

    if test.empty:
        raise ValueError("No autobahn stations found")
    if controll.empty:
        raise ValueError("No non autobahn stations found")
        
    # to avoid érrors, make sure k doesnt exceed number of available controll stations
    k_eff = min(k, len(controll))

    #combine lat and lng into one column
    test_xy = np.column_stack([test.geometry.x, test.geometry.y])
    controll_xy = np.column_stack([controll.geometry.x, controll.geometry.y])

    # build tree
    tree = KDTree(controll_xy)
    distances, indices = tree.query(test_xy, k = k_eff)

    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
 
    rows = []

    #for each autobahn station, query the tree to get the closest non autobahn stations
    for i in range(len(test)):
        test_id = test.loc[i, "uuid"]

        #add respective autobahn station to our list
        rows.append({
                "match_set_uuid" : test_id,
                "station_uuid": test_id,
                "autobahn": 1,
                "dist_km": 0.0
            })

        for dist, idx in zip(distances[i], indices[i]):
            dist_km = float(dist) / 1000.0 #convert from meters in km

            if max_dist is not None and dist_km > max_dist:
                continue

            rows.append({
                    "match_set_uuid": test_id,
                    "station_uuid": controll.loc[int(idx), "uuid"],
                    "autobahn": 0,
                    "dist_km": dist_km
                })
    match_map = pl.DataFrame(rows).unique(subset = ["match_set_uuid", "station_uuid"])

    #filter out invalid sets (see if there are enough controll points)
    valid_sets = (match_map
                      .group_by("match_set_uuid")
                      .agg([pl.len().alias("n"),
                            pl.col("autobahn").sum().alias("n_test")])
                      .filter(((pl.col("n") >= 2) & pl.col("n_test") == 1))
                      .select("match_set_uuid")                    
        )

    return match_map.join(valid_sets, on = "match_set_uuid", how = "inner")

# calculate mean absolute devianion
#written with the help of chatgpt
def __mad(x):
    x = np.asarray(x, dtype=float)
    median = np.median(x)
    return np.median(np.abs(x - median))


if __name__=="__main__":
   
    stations_path = Path(r'/Users/sebastian/data-science-projekt/tankerkoenig_data/stations/stations.csv')
    border_output_path = Path(r'/Users/sebastian/data-science-projekt/tankerkoenig_data/stations')
    #get_borderregion_stations(stations_path, border_output_path)
    
    #get_autobahn_stations(stations_path, border_output_path)
    filter_autobahn_from_borders(border_output_path / "lower_border_stations.csv", border_output_path / "autobahn_stations.csv", border_output_path)