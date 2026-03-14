from pathlib import Path
import polars as pl
import plotly.graph_objects as go
import pandas as pd
from linearmodels.iv.absorbing import AbsorbingLS


def display_weather_codes_per_region(weather_path:Path,weather_codes:Path,region:str,year:int):
    '''
    Displays a plotly scatterplot with the (hourly) weathercodes from a chosen region in a chosen year. Return nothing.
    i: Path weather_path, Path weather_codes, string, region, int year
    o: None
    '''

    file_path = weather_path / f"weather_region{region}.csv"

    df = pd.read_csv(file_path).drop(columns=["precipitation", "temperature_2m"])

    code_df = pd.read_csv(weather_codes)

    #downsmapling df to monthly avg.
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = pd.to_datetime(df["date"]).dt.year

    df_plot = df[df["year"] == year]

    # merge descriptions into the plot df.
    df_plot = df_plot.merge(code_df, on = "weather_code", how = "left")
    
    #sort weather codes by ascending order
    df_plot = df_plot.sort_values("weather_code", ascending=True)

    fig= go.Figure()

    # create scatter plot.
    fig.add_trace(
        go.Scatter(
            x = df_plot["date"],
            y = df_plot["weather_code"],
            mode = "markers",
            marker = dict(size=5,
                        color=df_plot["weather_code"],
                        colorscale="Turbo",
                        opacity=0.6),
            customdata = df_plot[["description"]],
            name = "",
            hovertemplate = (
                "Date: %{x|%Y-%m-%d %H:%M}<br>"
                "Weather code: %{y}<br>"
                "Description: %{customdata[0]}"
                "<extra></extra>"
            )

        )
    )
    fig.update_layout(
        title_text = f"weather codes for region {region} in year {year}",
        xaxis_title = "Date",
        yaxis_title = "Weather code",
        plot_bgcolor = "white",
        hovermode = "closest"
    )

    # fit for qualitative data (->discrete values)
    fig.update_yaxes(
        type = "category",
        showgrid = True,
        gridcolor = "lightgrey"
    )

    fig.update_xaxes(showgrid = True, gridcolor = "lightgrey")

    fig.show()

def run_volatility_panel_regression(price_path:Path,weather_path:Path,regions:list,extreme_weather_codes_file:Path,fuel_type:str="diesel",statistic:str="median",mad_window:int=24):
    '''
    Builds a panel dataset for each Leitregion and then estimates the effect of extreme weather on the gas stations price volatility.

    i: Path price_path, Path weather_path, list regions, Path extreme_weather_codes_file, string fuel_type, string statistic, int mad_window
    o: AbsorbingLSResults fixed-effects regression result
    '''

    price_column = f"{fuel_type}_{statistic}"

    #check if mad window is big enough.
    if mad_window < 2:
        raise ValueError("mad_window has to be at least 2.")

    #load the extreme weather codes.
    extreme_code_df = pd.read_csv(extreme_weather_codes_file)
    extreme_codes = extreme_code_df["Weather Codes"].tolist()

    #create an empty list for the regional polars dfs.
    regional_dfs = []

    #load and create the panels for each region.
    for region in regions:
        price_file = price_path / f"mean_median_price_region_{region}.csv"
        weather_file = weather_path / f"weather_region{region}.csv"

        #if one of the files is missing, skip the region.
        if not (price_file.exists() and weather_file.exists()):
            continue

        #use the polars lazy scan to load the data.
        price_lf = pl.scan_csv(price_file, try_parse_dates = True)
        weather_lf = pl.scan_csv(weather_file, try_parse_dates = True).select(["date", "weather_code"])

        #merge price and weather data.
        region_lf = (
            price_lf
            .join(
                weather_lf,
                left_on = "timestamp_utc",
                right_on = "date",
                how = "inner"
            )
            .sort("timestamp_utc")
            .with_columns([
                #create a column to indicate the region.
                pl.lit(region).alias("region"),

                #use a dummy variable for the weather codes (classify 1 as extreme, else 0).
                pl.col("weather_code").is_in(extreme_codes).cast(pl.Int8).alias("extreme_weather"),

                #time fixed effects (hourly and daily).
                pl.col("timestamp_utc").dt.date().alias("date"),
                pl.col("timestamp_utc").dt.hour().alias("hour"),

                #get the price changes (first difference) from hour to hour
                pl.col(price_column).diff().alias("price_change")
            ])
            .with_columns([
                #calculate the rolling mean of price changes over the selected window.
                pl.col("price_change").rolling_mean(window_size = mad_window, min_samples= mad_window).alias("rolling_mean_change")
            ])
            .with_columns([
                #calculater the absolute deviation from the local rolling mean.
                (pl.col("price_change") - pl.col("rolling_mean_change")).abs().alias("abs_deviation")
            ])
            .with_columns([
                #calculate mean absolute deviation of the price changes.
                pl.col("abs_deviation").rolling_mean(window_size = mad_window, min_samples = mad_window).alias("volatility")
            ])
            #drop rows where rolling mad is not available.
            .drop_nulls(["price_change", "rolling_mean_change", "abs_deviation", "volatility"])
        )
        regional_dfs.append(region_lf.collect())

    if not regional_dfs:
        raise ValueError("No valid regional dfs could be loaded.")
    
    #combine the regional dfs into one global (Germany) df.
    pl_panel_df = pl.concat(regional_dfs)

    #because linearmodels requires a pandas df, we convert our polars df into a pandas.
    panel_df = pl_panel_df.to_pandas()
    panel_df = panel_df.sort_values(["region", "timestamp_utc"]).reset_index(drop = True)

    #define the rolling mad of price changes as our dependent variable.
    y = panel_df["volatility"].astype(float)

    #define the extreme weather as our main explanatory variable.
    x = panel_df[["extreme_weather"]].astype(float)

    #define the regional and time fixed effects as factors the model has to absorb.
    absorbed_effects = pd.DataFrame(
        {
            "region_fe": panel_df["region"].astype("category"),
            "date_fe": panel_df["date"].astype("category"),
            "hour_fe": panel_df["hour"].astype("category")
        }
    )

    #cluster the standard error at region level.
    clusters = pd.DataFrame({
        "region_cluster": panel_df["region"].astype("category").cat.codes
    })

    #define the model (we use the AbsorbingLS model with the above defined variables)
    model = AbsorbingLS(
        dependent = y,
        exog = x,
        absorb = absorbed_effects,
        drop_absorbed = True
    )

    #fit the model.
    results = model.fit(
        cov_type = "clustered",
        clusters = clusters
    )

    #print the results and return them.
    print(results.summary)

    return results


