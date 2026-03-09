#With the help of CoPilot

import numpy as np
from sklearn.cluster import DBSCAN
import polars as pl
import plotly.express as px


def getCSVData(file_path):
    df = pl.read_csv(file_path, null_values=["nicht", "NA", "N/A", "", "Nicht"])
    print(df)
    return df

def performDBSCAN(data, eps, min_samples):
    cords = data.select(["latitude", "longitude"]).to_numpy()
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
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


df = getCSVData("C:\\Users\\Bjarne\\Desktop\\Uni\\Data Science Projekt\\PersonalTesting\\dbscan\\stations.csv")
labels = performDBSCAN(df, eps=0.01, min_samples=5)
plotClusters(df, labels)

