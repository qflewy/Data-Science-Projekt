import polars as pl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

fuel_labels = {"diesel_mean": "Diesel", "e5_mean": "E5", "e10_mean": "E10"}
fuel_cols = ["diesel_mean", "e5_mean", "e10_mean"]
fuel_label_list = ["Diesel", "E5", "E10"]
colours = ["steelblue", "darkorange", "seagreen"]


def add_rule_line(fig, rule_x, x_values=None, x_labels=None, **row_col):
    """
    Adds the red dashed line for the 12 PM rule (2026-04-01). On the yearly plots 2026 is split
    into two points (Jan-Mar at 2026.0, Apr-Oct at 2026.5), so the rule sits at 2026.25.
    If x_labels are given, every x value gets its own tick label (e.g. "2026<br>Apr-Oct").
    """
    if rule_x is None:
        return
    fig.add_vline(x=rule_x, line_color="red", line_dash="dash", line_width=2, **row_col)
    if x_values is not None and x_labels is not None:
        fig.update_xaxes(tickmode="array", tickvals=list(x_values), ticktext=list(x_labels), **row_col)


def plot_brand_vs_free_prices(brand, free, rule_x=None, x_labels=None):
    """
    Line chart with 3 subplots (Diesel, E5, E10) showing mean fuel prices
    for brand stations vs. free stations over the years.
    rule_x / x_labels: optional red line for the 12 PM rule, see add_rule_line.
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

        add_rule_line(fig, rule_x, brand["year"], x_labels, row=1, col=col_i)

    if rule_x is not None:
        fig.add_annotation(x=rule_x, y=1.0, yref="paper", xref="x", text="12 PM rule", showarrow=False,
                           xanchor="right", font=dict(color="red"))
    fig.update_yaxes(ticksuffix=" €", col=1)
    fig.update_layout(
        title="Mean Fuel Prices: Brand vs. Free Stations",
        height=450,
        width=1000,
        hovermode="x unified",
    )
    fig.show()


def plot_price_difference(brand, free, rule_x=None, x_labels=None):
    """
    Line chart showing the yearly price difference (brand minus free) in ct/L
    for each fuel type. A horizontal zero line marks the break-even point.
    With rule_x, the printed average only uses the full years before the rule and the
    points after it are printed separately.
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

        if rule_x is None:
            avg = sum(diff_ct) / len(diff_ct)
            print(f"{label}: avg difference = {avg:+.3f} ct/L")
        else:
            full_years = [d for x, d in zip(years, diff_ct) if x == int(x) and x < int(rule_x)]
            avg = sum(full_years) / len(full_years)
            later = ", ".join(f"{lab.replace('<br>', ' ')}: {d:+.3f}" for x, d, lab in zip(years, diff_ct, x_labels or years) if x >= int(rule_x))
            print(f"{label}: avg difference {int(years[0])}-{int(rule_x) - 1} = {avg:+.3f} ct/L | {later}")

    add_rule_line(fig, rule_x, years, x_labels)

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


def plot_brand_comparison(df_per_brand, df_brandvsfree, fuel="e10_mean", brands=None, rule_x=None, x_labels=None):
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
        pd.DataFrame(combined.to_dicts()),  # to_dicts instead of to_pandas, pyarrow is not installed
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
    add_rule_line(fig, rule_x, sorted(set(combined["year"].to_list())), x_labels)
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
        pd.DataFrame(avg_premium.to_dicts()),  # to_dicts instead of to_pandas, pyarrow is not installed
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

def period_means_from_monthly(df_monthly, group_col, rule_month=4):
    """
    2026 period values (before / after the 12 PM rule) as the mean of the monthly means.
    Pooling all updates of a period instead would overweight months in which a group reports
    more often; in March 2026 prices jumped by ~28 ct, so a brand with more March updates
    looked several ct more expensive than it was (e.g. HOYER).
    """
    return (
        df_monthly
        .with_columns(
            pl.when(pl.col("month") < rule_month).then(pl.lit("2026 Jan-Mar")).otherwise(pl.lit("2026 Apr-Oct")).alias("period")
        )
        .group_by([group_col, "period"])
        .agg([pl.col(c).mean() for c in fuel_cols])
        .sort([group_col, "period"])
    )


def plot_avg_premium_per_brand_periods(avg_old, per_brand_monthly, free_monthly, fuel="e10_mean", rule_month=4):
    """
    Grouped bar chart: average premium per brand vs. free stations for 2014-2025 (avg_old,
    columns brand_normalized + difference_ct) next to the two 2026 periods (before / after the 12 PM rule).
    For 2026 the premium is computed per month (brand vs. free in the same month) and then averaged.
    Sorted like the original chart (by the 2014-2025 premium).
    """
    free_col = "free_" + fuel
    new = (
        per_brand_monthly
        .join(free_monthly.select(["month", pl.col(fuel).alias(free_col)]), on="month")
        .with_columns([
            ((pl.col(fuel) - pl.col(free_col)) * 100).alias("difference_ct"),
            pl.when(pl.col("month") < rule_month).then(pl.lit("2026 Jan-Mar")).otherwise(pl.lit("2026 Apr-Oct")).alias("period"),
        ])
        .group_by(["brand_normalized", "period"])
        .agg(pl.col("difference_ct").mean())
    )
    old = avg_old.select(["brand_normalized", pl.lit("2014-2025 (avg)").alias("period"), "difference_ct"])
    order = avg_old.sort("difference_ct", descending=True)["brand_normalized"].to_list()
    combined = pl.concat([old, new.filter(pl.col("brand_normalized").is_in(order))])

    fig = px.bar(
        pd.DataFrame(combined.to_dicts()),
        x="brand_normalized",
        y="difference_ct",
        color="period",
        barmode="group",
        category_orders={"brand_normalized": order, "period": ["2014-2025 (avg)", "2026 Jan-Mar", "2026 Apr-Oct"]},
        color_discrete_map={"2014-2025 (avg)": "steelblue", "2026 Jan-Mar": "#b4b3ad", "2026 Apr-Oct": "red"},
        title=f"{fuel_labels[fuel]}: Brand vs. Free Station Average (red: since the 12 PM rule)",
        labels={"difference_ct": "Additional charge vs. free stations (ct/L)", "brand_normalized": "Brand", "period": "Period"},
    )
    fig.update_layout(height=500, width=1100, template="plotly_white")
    fig.add_hline(y=0, line_color="black", line_width=1)
    return fig, new


def plot_daily_gap_2026(daily, rule_date="2026-04-01", rebates=(("2026-05-01", "2026-07-01", "Fuel tax cut 1"),
                                                                ("2026-10-01", "2026-11-01", "Fuel tax cut 2"))):
    """
    Daily brand minus free price difference in 2026 (ct/L), from the 15-minute snapshot comparison
    (columns day, diesel_gap_ct, e5_gap_ct, e10_gap_ct). Red line: 12 PM rule, shaded: fuel tax cuts.
    """
    fig = go.Figure()
    for start, end, label in rebates:
        fig.add_vrect(x0=start, x1=end, fillcolor="steelblue", opacity=0.08, line_width=0,
                      annotation_text=label, annotation_position="top left")
    for i, fuel in enumerate(fuel_cols):
        col = fuel.replace("_mean", "_gap_ct")
        fig.add_trace(go.Scatter(
            x=daily["day"], y=daily[col], mode="lines", name=fuel_label_list[i],
            line=dict(color=colours[i], width=1.5),
            hovertemplate="%{x|%d.%m.}: %{y:+.2f} ct/L",
        ))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.add_vline(x=rule_date, line_color="red", line_dash="dash", line_width=2)
    fig.add_annotation(x=rule_date, y=1.0, yref="paper", text="12 PM rule", showarrow=False,
                       xanchor="right", font=dict(color="red"))
    fig.update_layout(
        title="Brand - Free per Day 2026 (snapshot comparison every 15 minutes)",
        xaxis_title="Date (2026)",
        yaxis_title="Difference (ct/L)",
        yaxis_ticksuffix=" ct",
        hovermode="x unified",
        height=450,
        width=1100,
        template="plotly_white",
    )
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