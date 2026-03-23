import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

fuel_labels = {"diesel_mean": "Diesel", "e5_mean": "E5", "e10_mean": "E10"}
fuel_cols = ["diesel_mean", "e5_mean", "e10_mean"]
fuel_label_list = ["Diesel", "E5", "E10"]
colours = ["steelblue", "darkorange", "seagreen"]


def plot_brand_vs_free_prices(brand, free):
    """
    Line chart with 3 subplots (Diesel, E5, E10) showing mean fuel prices
    for brand stations vs. free stations over the years.
    """
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True,
                        subplot_titles=fuel_label_list)

    for i in range(3):
        col = fuel_cols[i]
        col_i = i + 1
        show_legend = i == 0  # only show legend on first subplot to avoid duplicates

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
    """
    Line chart showing the yearly price difference (brand minus free) in ct/L
    for each fuel type. A horizontal zero line marks the break-even point.
    """
    years = brand["year"].to_list()

    fig = go.Figure()
    fig.add_hline(y=0, line_color="black", line_width=1)

    for i in range(3):
        col = fuel_cols[i]
        label = fuel_label_list[i]

        brand_vals = brand[col].to_list()
        free_vals = free[col].to_list()

        diff_ct = [(brand_vals[j] - free_vals[j]) * 100 for j in range(len(years))]  # convert €/L → ct/L

        fig.add_trace(go.Scatter(
            x=years,
            y=diff_ct,
            mode="lines+markers",
            name=label,
            line=dict(color=colours[i], width=2),
            hovertemplate="%{x}: %{y:+.2f} ct/L",
        ))

        avg = sum(diff_ct) / len(diff_ct)
        print(f"{label}: avg difference = {avg:+.3f} ct/L")

    fig.update_layout(
        title="Price Difference: Brand - Free",
        xaxis_title="Year",
        yaxis_title="Difference (ct/L)",
        yaxis_ticksuffix=" ct",
        hovermode="x unified",
        height=450,
        width=900,
    )
    fig.show()


def plot_brand_comparison(df_per_brand, df_brandvsfree, fuel="e10_mean", brands=None):
    """
    Line chart comparing yearly mean prices of selected individual brands
    against the free-station average for a single fuel type.
    """
    if brands is None:
        brands = ["ARAL", "OIL!", "ESSO"]

    subset = df_per_brand.filter(pl.col("brand_normalized").is_in(brands)).sort("year")
    free_line = (
        df_brandvsfree
        .filter(pl.col("station_type") == "free_station")
        .select(["year", fuel])
        .with_columns(pl.lit("Free Station (avg)").alias("brand_normalized"))  # give free avg a label so it shows up in the color legend
    )

    combined = pl.concat([subset.select(["year", fuel, "brand_normalized"]), free_line])  # align columns before concat

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
    """
    Bar chart showing each brand's average price difference (in ct/L) over
    free stations across all years, sorted descending. Bars below zero
    mean the brand is on average cheaper than free stations.
    """
    free_col = "free_" + fuel

    joined = df_per_brand.join(
        free.select(["year", fuel]).rename({fuel: free_col}), on="year"  # rename to avoid column name collision after join
    )
    joined = joined.with_columns(
        ((pl.col(fuel) - pl.col(free_col)) * 100).fill_nan(None).alias("difference_ct")
        # fill_nan because ORLEN has a NaN value in 2018; convert to null to skip it
    )

    avg_premium = (
        joined
        .drop_nulls("difference_ct")
        .group_by("brand_normalized")
        .agg(pl.col("difference_ct").mean())
        .sort("difference_ct", descending=True)
    )

    fig = px.bar(
        avg_premium.to_pandas(),
        x="brand_normalized",
        y="difference_ct",
        title=f"{fuel_labels[fuel]}: Brand vs. Free Station Average",
        labels={"difference_ct": "Additional charge vs. free stations (ct/L)", "brand_normalized": "Brand"},
        )
    
    fig.update_layout(
        height=500,
        width=1100,
        yaxis=dict(rangemode="normal"),
        template="plotly_white"
    )
    fig.add_hline(y=0, line_color="black", line_width=1)
    return fig

def save_png(fig, img_name:Path, legend:bool=False):
    """
    This method saves plotly figures with high resolution (for the poster) to the given output path. Before saving the plot, the method adjusts the text to an appropriate size. If the plot has legend, set legend to True so it also adjusts the legends font size before saving. The chosen figure name should contain the suffix ".png". Returns nothing.
    i: plotly Figure fig, Path img_name, bool legend
    o: None
    """

    px_w = 3700
    px_h = 2250

    fig = fig.full_figure_for_development(warn = False)
    fig.update_layout(
        autosize = False,
        width = px_w/2,
        height = px_h,
        font = dict(size=94),
        title = dict(font = dict(size =100),
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
                font = dict(size = 85),
                y= .92,
                xanchor="left",
                itemsizing="constant",
                x=0
            ),
            #legend_title = None
        )

    fig.update_xaxes(tickfont = dict(size = 80), title_font = dict(size = 90))
    fig.update_yaxes(tickfont = dict(size = 80), title_font = dict(size = 90))

    img_path = Path(r'D:/Bilder') #change path to own pc
    fig.write_image(img_path/img_name,
                    width=px_w,
                    height=px_h,
                    scale=1)