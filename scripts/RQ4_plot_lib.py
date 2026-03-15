import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fuel_labels = {"diesel_mean": "Diesel", "e5_mean": "E5", "e10_mean": "E10"}
_FUEL_COLS = ["diesel_mean", "e5_mean", "e10_mean"]
_FUEL_LABEL_LIST = ["Diesel", "E5", "E10"]
_COLORS = ["steelblue", "darkorange", "seagreen"]


def plot_brand_vs_free_prices(brand, free):
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True,
                        subplot_titles=_FUEL_LABEL_LIST)

    for i in range(3):
        col = _FUEL_COLS[i]
        col_i = i + 1
        show_legend = i == 0

        fig.add_trace(go.Scatter(
            x=brand["year"],
            y=brand[col].round(4),
            mode="lines+markers",
            name="Brand station",
            line=dict(color="steelblue", width=2),
            legendgroup="brand",
            showlegend=show_legend,
            hovertemplate="%{x}: %{y:.3f} €/L",
        ), row=1, col=col_i)

        fig.add_trace(go.Scatter(
            x=free["year"],
            y=free[col].round(4),
            mode="lines+markers",
            name="Free station",
            line=dict(color="darkorange", width=2, dash="dash"),
            legendgroup="free",
            showlegend=show_legend,
            hovertemplate="%{x}: %{y:.3f} €/L",
        ), row=1, col=col_i)

    fig.update_yaxes(ticksuffix=" €", col=1)
    fig.update_layout(
        title="Mean Fuel Prices: Brand vs. Free Stations",
        height=450,
        width=1000,
        hovermode="x unified",
    )
    fig.show()


def plot_price_difference(brand, free):
    years = brand["year"].to_list()

    fig = go.Figure()
    fig.add_hline(y=0, line_color="black", line_width=1)

    for i in range(3):
        col = _FUEL_COLS[i]
        label = _FUEL_LABEL_LIST[i]

        brand_vals = brand[col].to_list()
        free_vals = free[col].to_list()

        diff_ct = [(brand_vals[j] - free_vals[j]) * 100 for j in range(len(years))]

        fig.add_trace(go.Scatter(
            x=years,
            y=diff_ct,
            mode="lines+markers",
            name=label,
            line=dict(color=_COLORS[i], width=2),
            hovertemplate="%{x}: %{y:+.2f} ct/L",
        ))

        avg = sum(diff_ct) / len(diff_ct)
        print(f"{label}: avg difference = {avg:+.3f} ct/L")

    fig.update_layout(
        title="Price Difference: Brand - Free (positive = brand is more expensive)",
        xaxis_title="Year",
        yaxis_title="Difference (ct/L)",
        yaxis_ticksuffix=" ct",
        hovermode="x unified",
        height=450,
        width=900,
    )
    fig.show()


def plot_brand_comparison(df_per_brand, df_brandvsfree, fuel="e10_mean", brands=None):
    if brands is None:
        brands = ["ARAL", "OIL!", "ESSO"]

    subset = df_per_brand.filter(pl.col("brand_normalized").is_in(brands)).sort("year")
    free_line = (
        df_brandvsfree
        .filter(pl.col("station_type") == "free_station")
        .select(["year", fuel])
        .with_columns(pl.lit("Free Station (avg)").alias("brand_normalized"))
    )

    combined = pl.concat([subset.select(["year", fuel, "brand_normalized"]), free_line])

    fig = px.line(
        combined.to_pandas(),
        x="year",
        y=fuel,
        color="brand_normalized",
        markers=True,
        title=f"{fuel_labels[fuel]}: {', '.join(brands)} vs. free stations",
        labels={"year": "Year", fuel: "Mean price (€/L)", "brand_normalized": "Brand"},
    )
    fig.update_layout(
        yaxis_ticksuffix=" €",
        hovermode="x unified",
        height=450,
        width=900,
    )
    return fig


def plot_avg_premium_per_brand(df_per_brand, free, fuel="e10_mean"):
    free_col = "free_" + fuel

    joined = df_per_brand.join(
        free.select(["year", fuel]).rename({fuel: free_col}), on="year"
    )
    joined = joined.with_columns(
        ((pl.col(fuel) - pl.col(free_col)) * 100).fill_nan(None).alias("premium_ct")
        # fill_nan because ORLEN has a NaN value in 2018; convert to null to skip it
    )

    avg_premium = (
        joined
        .drop_nulls("premium_ct")
        .group_by("brand_normalized")
        .agg(pl.col("premium_ct").mean())
        .sort("premium_ct", descending=True)
    )

    fig = px.bar(
        avg_premium.to_pandas(),
        x="brand_normalized",
        y="premium_ct",
        title=f"{fuel_labels[fuel]}: Brand vs. Free Station Average",
        labels={"premium_ct": "Additional charge vs. free stations (ct/L)", "brand_normalized": "Brand"},
    )
    fig.update_layout(
        height=500,
        width=1100,
        yaxis=dict(rangemode="normal"),
    )
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.show()
