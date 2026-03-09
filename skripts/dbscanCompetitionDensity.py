#With the help of CoPilot

import numpy as np
from sklearn.cluster import DBSCAN
import polars as pl
import plotly.express as px


def getCSVData(file_path):
    
    df = pl.read_csv(file_path, null_values=["nicht", "NA", "N/A", "", "Nicht"])
    df = df.select(["uuid", "latitude", "longitude"])
    print(df)
    return df

def performDBSCAN(data, eps, min_samples):
    cords = data.select(["latitude", "longitude"]).to_numpy()
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(cords)
    print(labels)
    return labels

def plotClusters(data, labels):

    # Clusterlabels zum DataFrame hinzufügen
    data = data.with_columns(pl.Series("cluster", labels))

    # Plotly arbeitet einfacher mit pandas
    pdf = data.to_pandas()

    fig = px.scatter_mapbox(
        pdf,
        lat="latitude",
        lon="longitude",
        color="cluster",
        hover_name="uuid",
        zoom=5,
        center={"lat": 51.1657, "lon": 10.4515},  # Deutschland Mittelpunkt
        height=700
    )

    fig.update_layout(mapbox_style="open-street-map")

    fig.show()


df = getCSVData("C:\\Users\\Bjarne\\Desktop\\Uni\\Data Science Projekt\\PersonalTesting\\dbscan\\stations.csv")
labels = performDBSCAN(df, eps=0.01, min_samples=5)
plotClusters(df, labels)

