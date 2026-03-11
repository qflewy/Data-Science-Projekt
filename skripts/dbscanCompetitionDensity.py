#With the help of CoPilot

import numpy as np
from sklearn.cluster import DBSCAN
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
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
    ).select(["station_uuid","day",mean_col,median_col])

    # optional Autobahnfilter
    if motorway_df is not None:

        daily_prices = daily_prices.join(
            motorway_df.lazy(),
            on="station_uuid",
            how="anti"
        )

    clusters = pl.scan_csv(cluster_csv).rename(
        {"cluster":"cluster_label"}
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
        df.group_by(["day","group"])
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

        subset = pdf[pdf["group"]==g]

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
    
    return motorway_clusters