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

    # Plotly arbeitet einfacher mit pandas
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


def plot_cluster_prices_lazy_full(parquet_files, cluster_csv, fuel="diesel", title=None):
    """
    RAM-effiziente Version der Cluster-Plot-Funktion mit LazyFrames.
    Beibehaltung aller bisherigen Features: Mean & Median, Cluster vs Noise, fuel wählbar.

    Parameters
    ----------
    parquet_files : list of str
        Liste der Parquet-Dateien für die Tagespreise.
    cluster_csv : str
        Pfad zur CSV-Datei mit cluster labels.
    fuel : str
        Kraftstoffart: "diesel", "e5", "e10"
    title : str
        Plot-Titel.
    """

    # 1️⃣ LazyFrame für Tagespreise aus Parquet
    daily_prices_lazy = pl.concat([pl.scan_parquet(f) for f in parquet_files])
    
    # 2️⃣ Nur die relevanten Spalten laden
    mean_col = f"{fuel}_mean"
    median_col = f"{fuel}_median"
    cols_needed = ["station_uuid", "day", mean_col, median_col]
    daily_prices_lazy = daily_prices_lazy.select(cols_needed)

    # 3️⃣ Cluster CSV als LazyFrame
    clusters_lazy = pl.scan_csv(cluster_csv).rename({"cluster": "cluster_label"})
    
    # 4️⃣ Join: station_uuid -> uuid
    df_joined = daily_prices_lazy.join(
        clusters_lazy,
        left_on="station_uuid",
        right_on="uuid",
        how="left"
    )
    
    # 5️⃣ Cluster vs Noise Gruppe
    df_joined = df_joined.with_columns(
        pl.when(pl.col("cluster_label") == -1)
        .then(pl.lit("noise"))
        .otherwise(pl.lit("cluster"))
        .alias("group")
    )
    
    # 6️⃣ Aggregation Mean & Median pro Tag & Gruppe (LazyFrame)
    daily_groups = (
        df_joined
        .group_by(["day", "group"])
        .agg([
            pl.col(mean_col).mean().alias("mean_price"),
            pl.col(median_col).mean().alias("median_price")
        ])
        .sort("day")
        .collect()  # erst jetzt Materialisierung
    )
    
    # 7️⃣ Zu Pandas für Plotly konvertieren
    df_plot = daily_groups.to_pandas()
    
    # 8️⃣ Plotly-Figure erstellen (Mean = solid, Median = dotted)
    fig = go.Figure()
    for g in df_plot['group'].unique():
        subset = df_plot[df_plot['group'] == g]
        fig.add_trace(go.Scatter(
            x=subset['day'],
            y=subset['mean_price'],
            mode='lines',
            name=f"{g} mean",
            line=dict(dash='solid')
        ))
        fig.add_trace(go.Scatter(
            x=subset['day'],
            y=subset['median_price'],
            mode='lines',
            name=f"{g} median",
            line=dict(dash='dot')
        ))
    
    # 9️⃣ Layout
    fig.update_layout(
        title=title or f"{fuel.capitalize()} Prices: Cluster vs Noise (Mean & Median)",
        xaxis_title="Date",
        yaxis_title=f"{fuel.capitalize()} Price (€)",
        template="plotly_white",
        hovermode="x unified"
    )
    
    return fig