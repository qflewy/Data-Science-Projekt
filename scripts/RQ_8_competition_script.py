# Debugged with the help of CoPilot

from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from math import radians, sin, cos, sqrt, atan2


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c


def haversine_metric(p1, p2):
    return haversine(p1[0], p1[1], p2[0], p2[1])


def getCSVData(file_path):
    df = pl.read_csv(file_path, null_values=["nicht", "NA", "N/A", "", "Nicht"])
    print(df)
    return df


def performDBSCAN(data, eps, min_samples):
    cords = data.select(["latitude", "longitude"]).to_numpy()
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=haversine_metric)
    labels = dbscan.fit_predict(cords)
    unique_labels = set(labels)
    num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    print(f"Number of clusters: {num_clusters}")
    print(labels)
    return labels


def plotClusters(data, labels):

    # Clusterlabels zum DataFrame hinzufügen
    data = data.with_columns(pl.Series("cluster", labels))
    pdf = data.to_pandas()

    # distinguish between noise (-1) and points in a cluster
    pdf["status"] = np.where(pdf["cluster"] == -1, "noise", "clustered")

    fig = px.scatter_mapbox(
        pdf,
        lat="latitude",
        lon="longitude",
        color="status",
        color_discrete_map={"noise": "red", "clustered": "green"},
        hover_name="uuid",
        zoom=5,
        center={"lat": 51.1657, "lon": 10.4515},  # Deutschland Mittelpunkt
        height=700
    )

    fig.update_layout(mapbox_style="open-street-map")
    return fig


def create_CSV_with_cluster_labels(data, labels, output_path):
    data = data.with_columns(pl.Series("cluster", labels))
    data.write_csv(output_path)


def join_labels_and_group(daily_prices, cluster_csv, cluster_name):
    """
    daily_prices: DataFrame mit station_uuid, day, diesel_mean, diesel_median
    cluster_csv: CSV mit station_uuid + cluster
    cluster_name: str, z.B. "eps1" oder "eps2"
    """
    clusters = pl.read_csv(cluster_csv)

    # Cluster-Labeling sauber umbenennen
    clusters = clusters.rename({"cluster": "cluster_label"})

    # Join
    df_joined = daily_prices.join(
        clusters,
        left_on="station_uuid",
        right_on="uuid",  # je nachdem ob CSV 'uuid' oder 'station_uuid' hat
        how="left"
    )

    # Gruppe: Noise vs Cluster
    df_joined = df_joined.with_columns(
        pl.when(pl.col("cluster_label") == -1)
        .then(pl.lit("noise"))
        .otherwise(pl.lit("cluster"))
        .alias("group")
    )

    # Kennzeichnung für Cluster-Parameter
    df_joined = df_joined.with_columns(
        pl.lit(cluster_name).alias("cluster_method")
    )

    return df_joined


def plot_cluster_prices(
        parquet_files,
        cluster_csv,
        fuel="diesel",
        motorway_df=None,
        title=None):

    mean_col = f"{fuel}_mean"
    median_col = f"{fuel}_median"

    # Lazy parquet scan
    daily_prices = pl.concat(
        [pl.scan_parquet(f) for f in parquet_files]
    ).select(["station_uuid", "day", mean_col, median_col])

    # optional Autobahnfilter
    if motorway_df is not None:

        daily_prices = daily_prices.join(
            motorway_df.lazy(),
            on="station_uuid",
            how="anti"
        )

    clusters = pl.scan_csv(cluster_csv).rename(
        {"cluster": "cluster_label"}
    )

    df = daily_prices.join(
        clusters,
        left_on="station_uuid",
        right_on="uuid",
        how="left"
    )

    df = df.with_columns(
        pl.when(pl.col("cluster_label") == -1)
        .then(pl.lit("noise"))
        .otherwise(pl.lit("cluster"))
        .alias("group")
    )

    result = (
        df.group_by(["day", "group"])
        .agg([
            pl.col(mean_col).mean().alias("mean_price"),
            pl.col(median_col).mean().alias("median_price")
        ])
        .sort("day")
        .collect()
    )

    pdf = result.to_pandas()

    fig = go.Figure()

    for g in pdf["group"].unique():

        subset = pdf[pdf["group"] == g]

        fig.add_trace(go.Scatter(
            x=subset["day"],
            y=subset["mean_price"],
            mode="lines",
            name=f"{g} mean"
        ))

        fig.add_trace(go.Scatter(
            x=subset["day"],
            y=subset["median_price"],
            mode="lines",
            name=f"{g} median",
            line=dict(dash="dot")
        ))

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified"
    )

    return fig


def analyse_motorway_clusters(cluster_csv, motorway_df):

    # Clusterlabels laden
    clusters = pl.read_csv(cluster_csv)

    # Join mit Autobahnstationen
    motorway_clusters = clusters.join(
        motorway_df,
        left_on="uuid",
        right_on="station_uuid",
        how="inner"
    )

    # Gesamtzahl Autobahnstationen im Clustering
    total_motorway = motorway_clusters.height

    # Noise
    motorway_noise = motorway_clusters.filter(pl.col("cluster") == -1).height

    # Cluster
    motorway_clustered = motorway_clusters.filter(pl.col("cluster") != -1).height

    print("Cluster file:", cluster_csv)
    print("Motorway stations total:", total_motorway)
    print("Motorway stations in clusters:", motorway_clustered)
    print("Motorway stations as noise:", motorway_noise)

    # return motorway_clusters


def plot_cluster_difference(
        parquet_files,
        cluster_csv,
        fuel="diesel",
        motorway_df=None,
        title=None):
   
    mean_col = f"{fuel}_mean"
    median_col = f"{fuel}_median"

    daily_prices = pl.concat([pl.scan_parquet(f) for f in parquet_files])\
        .select(["station_uuid", "day", mean_col, median_col])

    if motorway_df is not None:
        daily_prices = daily_prices.join(
            motorway_df.lazy(),
            on="station_uuid",
            how="anti"
        )

    clusters = pl.scan_csv(cluster_csv).rename({"cluster":"cluster_label"})
    df = daily_prices.join(
        clusters,
        left_on="station_uuid",
        right_on="uuid",
        how="left"
    )

    df = df.with_columns(
        pl.when(pl.col("cluster_label") == -1)
        .then(pl.lit("noise"))
        .otherwise(pl.lit("cluster"))
        .alias("group")
    )

    grouped = df.group_by(["day", "group"]).agg([
        pl.col(mean_col).mean().alias("mean_price"),
        pl.col(median_col).mean().alias("median_price")
    ]).sort("day").collect()

    pdf = grouped.to_pandas()
    pdf["day"] = pd.to_datetime(pdf["day"])

    pivot_mean = pdf.pivot(
        index="day",
        columns="group",
        values="mean_price"
        )
    pivot_median = pdf.pivot(
        index="day",
        columns="group",
        values="median_price"
        )

    diff_df = pd.DataFrame({
        "day": pivot_mean.index,
        "mean_diff": pivot_mean["cluster"] - pivot_mean["noise"],
        "median_diff": pivot_median["cluster"] - pivot_median["noise"]
    })

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=diff_df["day"], 
                   y=diff_df["mean_diff"], mode="lines", 
                   name="Mean Difference", 
                   line=dict(color="blue"))
                   )

    fig.add_trace(go.Scatter(x=diff_df["day"], y=diff_df["median_diff"],
                             mode="lines", name="Median Difference",
                             line=dict(color="red", dash="dot")))

    fig.update_layout(
        title=title or f"{fuel.capitalize()} Price Difference (Cluster - Noise)",
        xaxis_title="Date",
        yaxis_title=f"{fuel.capitalize()} Price Difference (€)",
        template="plotly_white",
        hovermode="x unified"
    )

    return fig, diff_df


def compute_cluster_counts_over_time(daily_prices, cluster_csv):
    """
    Berechnet pro Tag die Anzahl geclusterter und ungeclusterter Stationen 
    sowie den Anteil geclustert.
    """
    # Cluster-Labels laden
    clusters = pl.read_csv(cluster_csv).rename({"cluster": "cluster_label"})

    # Join auf daily_prices
    df = daily_prices.join(
        clusters,
        left_on="station_uuid",
        right_on="uuid",
        how="left"
    )

    # Gruppe: cluster vs noise
    df = df.with_columns(
        pl.when(pl.col("cluster_label") == -1)
        .then(pl.lit("noise"))
        .otherwise(pl.lit("cluster"))
        .alias("group")
    )

    # Tages-Counts berechnen
    counts = (
        df.group_by(["day", "group"])
          .agg(pl.count().alias("count"))
          .pivot(values="count", index="day", columns="group")
          .fill_null(0)
          .with_columns(
              (pl.col("cluster") + pl.col("noise")).alias("total_count"),
              (pl.col("cluster") / (pl.col("cluster") + pl.col("noise"))).alias("cluster_share")
          )
          .sort("day")
    )

    return counts


def plot_cluster_counts_over_time(counts, title="Cluster vs Noise Over Time"):
    """
    Plot the number of clustered vs noise stations over time, along with the
    cluster share.

    counts:
    pl.DataFrame mit Spalten:
    'day', 'cluster', 'noise', 'total_count', 'cluster_share'
    """
    df = counts.to_pandas()

    fig = go.Figure()

    # Anzahl geclusterter Stationen
    fig.add_trace(go.Scatter(
        x=df['day'],
        y=df['cluster'],
        mode='lines',
        name='Clustered stations',
        line=dict(color='green')
    ))

    # Anzahl ungeclusterter Stationen
    fig.add_trace(go.Scatter(
        x=df['day'],
        y=df['noise'],
        mode='lines',
        name='Noise stations',
        line=dict(color='red')
    ))

    # Anteil geclustert auf sekundärer Achse
    fig.add_trace(go.Scatter(
        x=df['day'],
        y=df['cluster_share'],
        mode='lines',
        name='Cluster share',
        line=dict(color='blue', dash='dot'),
        yaxis='y2'
    ))

    # Layout mit sekundärer Achse
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Number of stations",
        yaxis2=dict(
            title="Cluster share",
            overlaying="y",
            side="right",
            range=[0, 1]
        ),
        template="plotly_white",
        hovermode="x unified"
    )

    return fig


def plot_motorway_cluster_pies(cluster_csv1, cluster_csv2, motorway_df):

    clusters1 = pl.read_csv(cluster_csv1)
    clusters2 = pl.read_csv(cluster_csv2)

    mw1 = clusters1.join(
        motorway_df,
        left_on="uuid",
        right_on="station_uuid",
        how="inner"
    )

    mw2 = clusters2.join(
        motorway_df,
        left_on="uuid",
        right_on="station_uuid",
        how="inner"
    )

    clustered1 = mw1.filter(pl.col("cluster") != -1).height
    noise1 = mw1.filter(pl.col("cluster") == -1).height

    clustered2 = mw2.filter(pl.col("cluster") != -1).height
    noise2 = mw2.filter(pl.col("cluster") == -1).height

    fig = go.Figure()

    fig.add_trace(go.Pie(
        labels=["Cluster", "Noise"],
        values=[clustered1, noise1],
        marker=dict(colors=["green", "red"]),
        textinfo="label+value",
        domain={"x": [0.0, 0.45], "y": [0, 1]},
        name="Cluster Set 1"
    ))

    fig.add_trace(go.Pie(
        labels=["Cluster", "Noise"],
        values=[clustered2, noise2],
        marker=dict(colors=["green", "red"]),
        textinfo="label+value",
        domain={"x": [0.55, 1.0], "y": [0, 1]},
        name="Cluster Set 2"
    ))

    fig.update_layout(
        title="Motorway Stations in Clusters",
        annotations=[
            dict(text="Cluster Set 1",
                 x=0.22, y=1.1, showarrow=False,
                 font=dict(size=16)),
            dict(text="Cluster Set 2",
                 x=0.78,
                 y=1.1,
                 showarrow=False,
                 font=dict(size=16))
        ]
    )

    return fig


def plot_yearly_boxplot(diff_df, cluster_name, fuel):
    # Jahr extrahieren
    diff_df["year"] = diff_df["day"].dt.year

    fig = go.Figure()

    for y in sorted(diff_df["year"].unique()):
        fig.add_trace(go.Box(
            y=diff_df.loc[diff_df["year"] == y, "mean_diff"],
            name=str(y),
            marker_color="blue",
            showlegend=False
        ))

    fig.update_layout(
        title=f"{fuel.capitalize()} Mean Price Difference by Year ({cluster_name})",
        xaxis_title="Year",
        yaxis_title="Price Difference (€) (Cluster - Noise)",
        template="plotly_white"
    )

    return fig


def save_png(fig, img_name:Path, legend:bool=False):

    px_w = 3000
    px_h = 2250

    fig = fig.full_figure_for_development(warn = False)
    fig.update_layout(
        autosize = False,
        width = px_w/2,
        height = px_h,
        font = dict(size=44),
        title = dict(font = dict(size =50),
                     y = .98,
                     x = .5,
                     xanchor = "center",
                     yanchor = "top")
    )
    if legend:
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                font = dict(size = 38),
                y= .92,
                xanchor="left",
                x=0
            )
        )

    fig.update_xaxes(tickfont = dict(size = 38), title_font = dict(size = 40))
    fig.update_yaxes(tickfont = dict(size = 38), title_font = dict(size = 40))
    img_path = Path(r'C:\Users\Bjarne\Desktop\Uni\Data Science Projekt\images_for_poster')
    
    # Wir extrahieren nur den Dateinamen (.name), um doppelten Pfad-Salat zu vermeiden
    # Und wir konvertieren das gesamte Path-Objekt am Ende in einen String
    final_file = img_path / Path(img_name).name
    
    fig.write_image(str(final_file),
                    width=px_w,
                    height=px_h,
                    scale=1)