from pathlib import Path
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import pandas as pd
import polars.selectors as cs
from IPython.display import display
from linearmodels.iv.absorbing import AbsorbingLS

def display_weather_per_region(weather_path:Path,region:str,year:str):
    
    file_path = weather_path / f"weather_region{region}.csv"
    print(f"Reading file from: {file_path}")

    df = pd.read_csv(file_path).drop(columns=["weather_code"])
    print("df successfully created.")

    #downsmapling df to monthly avg
    print("Downsampling df...")
    df["date"] = pd.to_datetime(df["date"])

    df.set_index("date", inplace=True)
    df_plot = df.loc[year]

    # create plotly schema with two y axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    #add left axis for temperature
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot["temperature_2m"],
        name="Temperature (°C)",
        mode="lines",
        line=dict(color="firebrick", width=2),
        hovertemplate="%{y:.1f} °C"
        ),
        secondary_y=False
    )

    #add right axis for precipitaion
    fig.add_trace(go.Bar(
        x=df_plot.index,
        y=df_plot["precipitation"],
        name="Precipitation (mm)",
        marker_color="cornflowerblue",
        opacity=0.5,
        hovertemplate="%{y:.1f} mm"
        ),
        secondary_y=True,
    )

    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot["precipitation"],
        name="Precipitation (trend)",
        mode="lines+markers",
        line=dict(color="darkblue", width=2),
        hovertemplate="%{y:.1f} mm"
        ),
        secondary_y=True,
    )

    #make sure both y lines have same 0 point
    t_min = min(df_plot["temperature_2m"].min(),0)
    t_max = max(df_plot["temperature_2m"].max(),1)
    p_max = max(df_plot["precipitation"].max(), 1)

    t_max = t_max*1.1
    p_max = p_max*1.1
    if t_min < 0:
        t_min = t_min*1.1
        
        p_min_new = t_min * (p_max / t_max)
    else:
        p_min_new = 0

    fig.update_yaxes(range=[t_min, t_max], secondary_y=False)
    fig.update_yaxes(range=[p_min_new, p_max], secondary_y=True)
    fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")

    #fit layout (title, hovering, legend)
    fig.update_layout(
        title_text= f"Clima diagramm for year {year} for region {region}",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white"
    )

    #name and color axis
    fig.update_xaxes(showgrid=True, gridcolor="lightgrey")
    fig.update_yaxes(title_text="<b>Temperature</b> (°C)", color="firebrick", secondary_y=False)
    fig.update_yaxes(title_text="<b>Precipitation</b> (mm)", color="darkblue", secondary_y=True, showgrid=False)

   
    
    fig.show()


def display_weather_codes_per_region(weather_path:Path,weather_codes:Path,region:str,year:int):

    file_path = weather_path / f"weather_region{region}.csv"

    df = pd.read_csv(file_path).drop(columns=["precipitation", "temperature_2m"])

    code_df = pd.read_csv(weather_codes)

    #downsmapling df to monthly avg
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = pd.to_datetime(df["date"]).dt.year

    df_plot = df[df["year"] == year]

    # merge descriptions into the plot df
    df_plot = df_plot.merge(code_df, on = "weather_code", how = "left")
    
    #sort weather codes by ascending order
    df_plot = df_plot.sort_values("weather_code", ascending=True)

    fig= go.Figure()

    # create scatter plot
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
            name = "weather_code",
            hovertemplate = (
                "Date: %{x}<br>"
                "Weather code: %{y}<br>"
                "Description: %{customdata[0]}"
            )

        )
    )

    #fit layout
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

#NOTE: not in use anymore due to faulty test assumptions
def levene_test_for_extreme_weather(weather_path:Path, region_price_path:Path, region:str):
    
    weather_file = weather_path / f'weather_region{region}.csv'
    price_file = region_price_path / f'mean_median_price_region_{region}.csv'

    if price_file.exists() and weather_file.exists():
       pass
    else:
        return [{"region": region, "variable": "all", "significant": None, "note": "missing files"}]

    try:

        weather_df = pl.read_csv(weather_file, try_parse_dates=True)
        price_df = pl.read_csv(price_file, try_parse_dates=True)

        # merge all rows with matching timestamps
       
        df_merged = (price_df.join(weather_df,
                                left_on="timestamp_utc", 
                                right_on="date",
                                how="inner")
                                .select(["timestamp_utc", "diesel_median", "e5_median", "e10_median", "weather_code"]))
        
        df_merged = df_merged.sort(["timestamp_utc"])

        # retrieve price differences for stationary data
        df_merged = df_merged.with_columns(pl.col(["diesel_median", "e5_median", "e10_median"])
                                        .diff().name.suffix("_diff"))
        
        #change path for other user
        extreme_codes_df = pd.read_csv(Path(r'/Users/sebastian/data-science-projekt/extreme_weather_codes_copy.csv'))
        extreme_codes = extreme_codes_df["Weather Codes"].to_list()

        
        df_extreme = (
            df_merged.filter(pl.col("weather_code").is_in(extreme_codes))
            .select(["timestamp_utc"])
            .rename({"timestamp_utc": "event_time"})
        )
        
        #security check if there was extreme weather
        if df_extreme.height == 0:
            print(f"no extreme weather for region {region}")
            return [{"region": region, "variable": "all", "test_statistic": None, "p_value": None,"significant": None, "note": "No extreme weather"}]
        
        # find and join nearest event
        df_joined = df_merged.join_asof(
            df_extreme,
            left_on="timestamp_utc",
            right_on="event_time",
            strategy="nearest"
        )

        #define time interval
        df_analysis = df_joined.with_columns(
            (pl.col("timestamp_utc") - pl.col("event_time")).dt.total_days().abs().alias("days_to_event")
        )

        # event interval: +/- 3 days | controll interval: >7 days from event
        event_mask = pl.col("days_to_event") <= 1
        controll_mask = pl.col("days_to_event") > 7

        df_event = df_analysis.filter(event_mask)
        df_controll = df_analysis.filter(controll_mask)

        #select only columns with the differences
        diff_spalten = df_analysis.select(cs.ends_with("_diff")).columns

        # create dicts with independent numpy arrays for event and controll group
        event_arrays = {
            col : df_event.drop_nulls(col)[col].to_numpy()
            for col in diff_spalten
        }
        controll_arrays = {
            col : df_controll.drop_nulls(col)[col].to_numpy()
            for col in diff_spalten
        }

        # statistic test for each column (looped)
        
        results_region = []

        for col in diff_spalten:
            arr_e = event_arrays[col]
            arr_c = controll_arrays[col]

            #clean up column name
            variable_name = col.replace("_diff", "")

            #check if there is enough data for this column
            if len(arr_e) < 2 or len(arr_c) < 2:
                
                results_region.append({
                    "region": region,
                    "variable": variable_name,
                    "test_statistic": None,
                    "p_value": None,
                    "significant": None,
                    "note": "not enough data points"
                })
                continue

            #Levene test (tests the equality of the variance of two populations) 
            #TODO: add latex description to notebook (https://de.wikipedia.org/wiki/Levene-Test)
            stat, p_value = stats.levene(arr_e, arr_c)

            results_region.append({
                "region": region,
                "variable": variable_name,
                "test_statistic": stat,
                "p_value": p_value,
                "significant": bool(p_value < .05),
                "data_event": len(arr_e),
                "data_controll": len(arr_c),
                "note": "Success!"
            })
        
        
        return results_region
        
    except Exception as e:
        return [{"region": region, "significant": None, "variable": "error", "note": str(e)}]
#NOTE: not in use anymore due to faulty test assumptions
def display_levene_test_results(results_df:pd.DataFrame):
    formated_table = (
        results_df.style
        .format({
            "test_statistic": "{:.2f}",
            "p_value": "{:.4f}",
           
        })
        .map(__highlight_significance, subset=["significant"])
        .set_caption("Levene test: Influence of extreme weather events on price variance")
    )

    display(formated_table)
#NOTE: not in use anymore due to faulty test assumptions
def __highlight_significance(val):
    if pd.isna(val) or isinstance(val, str):
        return ""
    color = "lightgreen" if val == True else "transparent"
    return f"background-color: {color}"
#NOTE: not in use anymore due to faulty test assumptions
def extract_arrays_for_global_test(weather_path:Path, region_price_path:Path, region:str):
    weather_file = weather_path / f'weather_region{region}.csv'
    price_file = region_price_path / f'mean_median_price_region_{region}.csv'

    if price_file.exists() and weather_file.exists():
       pass
    else:
        return {"region": region, "status": "Files not ok", "arrays": {}}
    
    try:

        weather_df = pl.read_csv(weather_file, columns=["date", "weather_code"], try_parse_dates=True)
        price_df = pl.read_csv(price_file, try_parse_dates=True)

        # merge all rows with matching timestamps
       
        df_merged = (price_df.join(weather_df,
                                left_on="timestamp_utc", 
                                right_on="date",
                                how="inner")
                                .select(["timestamp_utc", "diesel_median", "e5_median", "e10_median", "weather_code"]))
        
        df_merged = df_merged.sort(["timestamp_utc"])

        # retrieve price differences for stationary data
        df_merged = df_merged.with_columns(pl.col(["diesel_median", "e5_median", "e10_median"])
                                        .diff().name.suffix("_diff"))
        
        #change path for other user
        extreme_codes_df = pd.read_csv(Path(r'/Users/sebastian/data-science-projekt/extreme_weather_codes.csv'))
        extreme_codes = extreme_codes_df["Weather Codes"].to_list()

        
        df_extreme = (
            df_merged.filter(pl.col("weather_code").is_in(extreme_codes))
            .select(["timestamp_utc"])
            .rename({"timestamp_utc": "event_time"})
        )
        
        #security check if there was extreme weather
        if df_extreme.height == 0:
            print(f"no extreme weather for region {region}")
            return {"region": region, "status": "no extreme weather", "arrays": {}}
        
        # find and join nearest event
        df_joined = df_merged.join_asof(
            df_extreme,
            left_on="timestamp_utc",
            right_on="event_time",
            strategy="nearest"
        )

        #define time interval
        df_analysis = df_joined.with_columns(
            (pl.col("timestamp_utc") - pl.col("event_time")).dt.total_days().abs().alias("days_to_event")
        )

        # event interval: +/- 3 days | control interval: >7 days from event
        event_mask = pl.col("days_to_event") <= 1
        controll_mask = pl.col("days_to_event") > 7

        df_event = df_analysis.filter(event_mask)
        df_controll = df_analysis.filter(controll_mask)

        #select only columns with the differences
        diff_spalten = df_analysis.select(cs.ends_with("_diff")).columns

        # collecting arrays into an dict:
        collected_arrays = {}
        for col in diff_spalten:
            var_name = col.replace("_diff", "")
            arr_e = df_event.drop_nulls(col)[col].to_numpy()
            arr_c = df_controll.drop_nulls(col)[col].to_numpy()

            #only add if they contain data
            if len(arr_e) >= 2 and len(arr_c) >= 2:
                collected_arrays[var_name] = {"event":arr_e, "control": arr_c}

        return {"region": region, "status": "Success", "arrays": collected_arrays}
    
    except Exception as e:
        return {"region": region, "status": f"Error: {str(e)}", "arrays": {}}
#NOTE: not in use anymore due to faulty test assumptions
def calculate_global_levene(global_arrays:dict):

    global_results = []

    for var_name, data in global_arrays.items():
        #check if there is data to concat
        if len(data["event"]) == 0 or len(data["control"]) == 0:
            global_results.append({
                "variable": var_name,
                "test_statistic": None,
                "p_value": None,
                "significant": None,
                "note": "Missing data for global test"
            })
            continue
        
        #put all regional arrays into global array
        arr_event_global = np.concatenate(data["event"])
        arr_controll_global = np.concatenate(data["control"])

        # calculate global levene
        stat, p_value = stats.levene(arr_event_global, arr_controll_global)

        # safe results in format for display method
        global_results.append({
            "variable": var_name,
            "test_statistic": stat,
            "p_value": p_value,
            "significant": bool(p_value < 0.05),
            "data_event": len(arr_event_global),
            "data_controll": len(arr_controll_global),
            "note": "Global Success!"
        })
    return global_results

""" #TODO: falls methode weiter genutzt werden soll, muss sie an standards angepasst werden
#aktuell nur kurz zu testzwecken eingefügt!!
#aktuell testet es nur e10!!!
def run_volatility_panel_regression(ordner_preise:Path,ordner_wetter: Path, regionen: list):
    """
    #Erstellt ein riesiges Panel aus allen Regionen und schätzt den Effekt 
    #von Extremwetter auf die Preisvolatilität via Fixed Effects Regression.
"""
    df_list = []
    
    extreme_codes_df = pd.read_csv(Path(r'/Users/sebastian/data-science-projekt/extreme_weather_codes_copy.csv'))
    extreme_codes =  extreme_codes_df["Weather Codes"].to_list()

    print("1. Lade und verarbeite regionale Daten für das Panel...")
    for region in regionen:
        preis_datei = ordner_preise / f"mean_median_price_region_{region}.csv"
        wetter_datei = ordner_wetter / f"weather_region{region}.csv"

        if preis_datei.exists() and wetter_datei.exists():
            # Wir nutzen LazyFrames (scan_csv), damit Polars extrem RAM-schonend arbeitet
            lf_preis = pl.scan_csv(preis_datei, try_parse_dates=True)
            lf_wetter = pl.scan_csv(wetter_datei, try_parse_dates=True).select(["date", "weather_code"])

            lf = (
                lf_preis.join(lf_wetter, left_on="timestamp_utc", right_on="date", how="inner")
                .sort("timestamp_utc")
                .with_columns([
                    # Füge die Region als Spalte hinzu (Wichtig für das Panel!)
                    pl.lit(region).alias("region"),
                    
                    # VOLATILITÄT: Absoluter Betrag der Preisdifferenz
                    pl.col("e10_median").diff().abs().alias("volatilitaet"),
                    
                    # DUMMY: 1 wenn Extremwetter, sonst 0
                    pl.col("weather_code").is_in(extreme_codes).cast(pl.Int8).alias("extremwetter"),
                    
                    # FIXED EFFECTS DIMENSIONEN: Datum und Stunde extrahieren
                    pl.col("timestamp_utc").dt.date().alias("datum"),
                    pl.col("timestamp_utc").dt.hour().alias("stunde")
                ])
                # Regressionen hassen Null-Werte, also werfen wir die erste leere Diff-Zeile weg
                .drop_nulls("volatilitaet") 
            )
            # Führt den Lazy-Plan aus und packt das DataFrame in die Liste
            df_list.append(lf.collect()) 

    print("2. Setze Panel-Datensatz zusammen...")
    # Klebt alle Regionen untereinander
    df_panel = pl.concat(df_list)

    print(f"Panel erstellt! {df_panel.height:,} Zeilen. Konvertiere für pyfixest...")
    # pyfixest braucht ein Pandas DataFrame
    df_pandas = df_panel.to_pandas()

    print("3. Berechne Fixed Effects Regression & Clustered Standard Errors...")
    # DIE REGRESSION
    # Syntax: Abhängige_Variable ~ Unabhängige_Variable | Fixed_Effects
    # vcov={'CRV1': 'region'} -> Clustered Standard Errors auf Regionsebene!
    modell = feols(
        "volatilitaet ~ extremwetter | region + datum + stunde", 
        data=df_pandas, 
        vcov={'CRV1': 'region'}
    )

    print("\n" + "="*50)
    print(" ERGEBNIS DER PANEL-REGRESSION")
    print("="*50)
    # Zeigt die wissenschaftliche Summary-Tabelle an
    modell.summary()

    return modell

 """

def run_volatility_panel_regression(price_path:Path,weather_path:Path,regions:list,extreme_weather_codes_file:Path,fuel_type:str="diesel",statistic:str="median",mad_window:int=24):
    '''
    Builds a panel dataset for each Leitregion and then estimates the effect of extreme weather on the gas stations price volatility.

    i: Path price_path, Path weather_path, list regions, Path extreme_weather_codes_file, string fuel_type, string statistic, int mad_window
    o: AbsorbingLSResults fe regression result
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
    print("\n" + "=" * 60)
    print("PANEL REGRESSION RESULTS")
    print("=" * 60)
    print(results.summary)

    return results


