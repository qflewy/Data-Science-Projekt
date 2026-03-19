from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ipywidgets as widgets
import re
import polars as pl
from IPython.display import display, Markdown, clear_output


def fuel_surface_dashboard(data_dir, stat="median", timezone="Europe/Berlin"):
    data_dir = Path(data_dir)

    region_files = {
        f"{i:02d}": data_dir / f"mean_median_price_region_{i:02d}.csv"
        for i in range(1, 100)
        if (data_dir / f"mean_median_price_region_{i:02d}.csv").exists()
    }

    weekday_labels = ["Mon.", "Tue.", "Wed.", "Thu.", "Fri.", "Sat.", "Sun."]

    def load_region(region):
        df = pd.read_csv(region_files[region])
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df["timestamp"] = (
            df["timestamp_utc"]
            .dt.tz_convert(timezone)
            .dt.tz_localize(None)
        )
        df["week"] = df["timestamp"].dt.to_period("W").dt.start_time
        return df

    def build_matrix(df, fuel, period):
        col = f"{fuel}_{stat}"

        df = df[df["week"] == period].copy()
        df["weekday"] = df["timestamp"].dt.weekday
        df["hour"] = df["timestamp"].dt.hour

        grouped = (
            df.groupby(["hour", "weekday"], as_index=False)[col]
            .mean()
            .rename(columns={col: "price"})
        )

        pivot = (
            grouped.pivot(index="hour", columns="weekday", values="price")
            .reindex(index=range(24), columns=range(7))
        )

        price_matrix = pivot.to_numpy(dtype=float)

        if np.all(np.isnan(price_matrix)):
            min_pos = None
        else:
            idx = np.nanargmin(price_matrix)
            r, c = np.unravel_index(idx, price_matrix.shape)
            min_pos = {
                "weekday": c,
                "hour": r,
                "price": price_matrix[r, c],
            }

        hover = np.empty(price_matrix.shape, dtype=object)

        for h in range(24):
            for w in range(7):
                val = price_matrix[h, w]
                if np.isnan(val):
                    hover[h, w] = (
                        f"Weekday: {weekday_labels[w]}<br>"
                        f"Hour: {h:02d}:00<br>"
                        f"Price: no data"
                    )
                else:
                    hover[h, w] = (
                        f"Weekday: {weekday_labels[w]}<br>"
                        f"Hour: {h:02d}:00<br>"
                        f"Price: {val:.3f}"
                    )

        return price_matrix, hover, min_pos

    region_dd = widgets.Dropdown(
        options=sorted(region_files.keys()),
        value=sorted(region_files.keys())[0],
        description="Region",
    )

    fuel_dd = widgets.Dropdown(
        options=["diesel", "e5", "e10"],
        value="diesel",
        description="Fueltype",
    )

    df0 = load_region(region_dd.value)
    weeks = sorted(df0["week"].unique())

    slider = widgets.SelectionSlider(
        options=[(str(w.date()), w) for w in weeks],
        value=weeks[0],
        description="Week",
        continuous_update=False,
        layout=widgets.Layout(width="900px"),
    )

    z, hover, min_pos = build_matrix(df0, fuel_dd.value, slider.value)

    fig = go.FigureWidget()
    fig.add_trace(
        go.Surface(
            x=list(range(7)),          # weekday
            y=list(range(24)),         # hour
            z=z,                       # shape (24,7) = [hour, weekday]
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            colorbar=dict(title="Price"),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode="markers+text",
            marker=dict(size=6, color="red"),
            name="Minimum",
            hovertemplate="%{text}<extra></extra>",
            text=[],
        )
    )
    fig.update_layout(
        width=1100,
        height=800,
        scene=dict(
            xaxis=dict(
                title="Weekday",
                tickvals=list(range(7)),
                ticktext=weekday_labels,
            ),
            yaxis=dict(
                title="Hour",
                tickvals=list(range(0, 24, 2)),
            ),
            zaxis=dict(
                title="Price",
                autorange="reversed",
            ),
        ),
    )

    def update(*args):
        df = load_region(region_dd.value)

        weeks_new = sorted(df["week"].unique())
        slider.options = [(str(w.date()), w) for w in weeks_new]
        if slider.value not in weeks_new:
            slider.value = weeks_new[0]

        z, hover, min_pos = build_matrix(df, fuel_dd.value, slider.value)

        with fig.batch_update():
            fig.data[0].z = z
            fig.data[0].text = hover

            if min_pos is not None:
                fig.data[1].x = [min_pos["weekday"]]
                fig.data[1].y = [min_pos["hour"]]
                fig.data[1].z = [min_pos["price"] - 1e-6]
                fig.data[1].text = [[
                    f"cheapest point<br>"
                    f"Week-day: {weekday_labels[min_pos['weekday']]}<br>"
                    f"Hour: {min_pos['hour']:02d}:00<br>"
                    f"Price: {min_pos['price']:.3f}"
                ]]
            else:
                fig.data[1].x = []
                fig.data[1].y = []
                fig.data[1].z = []
                fig.data[1].text = []

    region_dd.observe(update, names="value")
    fuel_dd.observe(update, names="value")
    slider.observe(update, names="value")

    ui = widgets.VBox([
        widgets.HBox([region_dd, fuel_dd]),
        slider,
        fig
    ])
    return ui


# ----------------------------------------------------------


def tank_time_analysis_dashboard(
    data_dir: str | Path = r"D:/data/derived/station_price_observations_web",
    file_pattern: str = "*.parquet",
):
    data_dir = Path(data_dir)

    files = sorted(data_dir.glob(file_pattern))
    month_pattern = re.compile(r"(\d{4})_(\d{2})")

    month_files: dict[str, Path] = {}
    for f in files:
        m = month_pattern.search(f.stem)
        if m:
            month_key = f"{m.group(1)}-{m.group(2)}"
            month_files[month_key] = f

    if not month_files:
        raise FileNotFoundError(f"Found no matching data files in {data_dir}.")

    month_keys = sorted(month_files.keys())

    month_slider = widgets.SelectionRangeSlider(
        options=month_keys,
        index=(0, len(month_keys) - 1),
        description="Time Range",
        layout=widgets.Layout(width="900px"),
        continuous_update=False,
    )

    fuel_dropdown = widgets.Dropdown(
        options=["diesel", "e5", "e10"],
        value="diesel",
        description="Fuel",
    )

    stat_dropdown = widgets.Dropdown(
        options=[("Mean", "mean"), ("Median", "median")],
        value="mean",
        description="Statistic",
    )

    city_text = widgets.Text(
        value="",
        description="City",
        placeholder="eg. Hamburg",
    )

    brand_text = widgets.Text(
        value="",
        description="Brand",
        placeholder="eg. ARAL",
    )

    plz_text = widgets.Text(
        value="",
        description="PLZ",
        placeholder="eg. 24",
    )

    output = widgets.Output()

    weekday_labels = ["Mon.", "Tue.", "Wed.", "Thu.", "Fri.", "Sat.", "Sun."]

    def load_period_lazy(start_month: str, end_month: str) -> pl.LazyFrame | None:
        selected_months = [m for m in month_keys if start_month <= m <= end_month]
        selected_files = [month_files[m] for m in selected_months]

        if not selected_files:
            return None

        lfs = [
            pl.scan_parquet(f).select([
                "station_id",
                "hour",
                "weekday",
                "fuel_type",
                "diff_to_min",
                "is_daily_min",
                "city_lc",
                "brand_lc",
                "post_code_str",
            ])
            for f in selected_files
        ]

        return pl.concat(lfs)

    def apply_filters_lazy(
        lf: pl.LazyFrame,
        fuel_type: str,
        city: str,
        brand: str,
        plz_prefix: str,
    ) -> pl.LazyFrame:
        out = lf.filter(pl.col("fuel_type") == fuel_type)

        city_clean = city.strip().lower()
        brand_clean = brand.strip().lower()
        plz_clean = plz_prefix.strip()

        if city_clean:
            out = out.filter(pl.col("city_lc").str.contains(city_clean, literal=True))

        if brand_clean:
            out = out.filter(pl.col("brand_lc").str.contains(brand_clean, literal=True))

        if plz_clean:
            out = out.filter(pl.col("post_code_str").str.starts_with(plz_clean))

        return out

    def compute_hour_stats(lf: pl.LazyFrame) -> pl.DataFrame:
        return (
            lf.group_by("hour")
            .agg([
                pl.mean("diff_to_min").alias("mean_diff"),
                pl.median("diff_to_min").alias("median_diff"),
                pl.mean("is_daily_min").alias("prob_min"),
            ])
            .sort("hour")
            .collect(engine="streaming")
        )

    def compute_heatmap_stats(lf: pl.LazyFrame) -> pl.DataFrame:
        return (
            lf.group_by(["weekday", "hour"])
            .agg([
                pl.mean("diff_to_min").alias("mean_diff"),
                pl.median("diff_to_min").alias("median_diff"),
            ])
            .sort(["weekday", "hour"])
            .collect(engine="streaming")
        )

    def compute_meta_stats(lf: pl.LazyFrame) -> dict:
        meta = (
            lf.select([
                pl.len().alias("n_obs"),
                pl.col("station_id").n_unique().alias("n_stations"),
            ])
            .collect(engine="streaming")
            .row(0, named=True)
        )
        return meta

    def build_summary_text(
        hour_stats: pl.DataFrame,
        meta: dict,
        start_month: str,
        end_month: str,
        fuel_type: str,
        city: str,
        brand: str,
        plz_prefix: str,
    ) -> str:
        if hour_stats.is_empty():
            return "No data available."

        best_hour_row = hour_stats.sort("mean_diff").row(0, named=True)
        best_hour = int(best_hour_row["hour"])
        best_hour_ct = float(best_hour_row["mean_diff"]) * 100
        best_prob = float(best_hour_row["prob_min"]) * 100

        evening = hour_stats.filter(pl.col("hour").is_between(18, 21, closed="both"))
        morning = hour_stats.filter(pl.col("hour").is_between(6, 9, closed="both"))

        evening_ct = float(evening.select(pl.mean("mean_diff")).item()) * 100 if evening.height > 0 else float("nan")
        morning_ct = float(morning.select(pl.mean("mean_diff")).item()) * 100 if morning.height > 0 else float("nan")
        diff_ct = morning_ct - evening_ct

        hour_map = {row["hour"]: row["mean_diff"] for row in hour_stats.to_dicts()}
        window_results = []

        for width in [2, 3, 4]:
            for start in range(0, 24 - width + 1):
                vals = [hour_map.get(h) for h in range(start, start + width)]
                if all(v is not None for v in vals):
                    avg = sum(vals) / width
                    window_results.append({
                        "width": width,
                        "start": start,
                        "end": start + width - 1,
                        "avg_diff": avg,
                    })

        best_window = min(window_results, key=lambda x: x["avg_diff"])
        best_window_ct = best_window["avg_diff"] * 100

        filters = [fuel_type]
        if city.strip():
            filters.append(f"city contains '{city.strip()}'")
        if brand.strip():
            filters.append(f"brand contains '{brand.strip()}'")
        if plz_prefix.strip():
            filters.append(f"postcode starts with '{plz_prefix.strip()}'")

        filter_text = ", ".join(filters)

        n_obs = int(meta["n_obs"])
        n_stations = int(meta["n_stations"])

        return (
            f"**Time period:** {start_month} to {end_month}  \n"
            f"**Filters:** {filter_text}  \n"
            f"**Observations:** {n_obs:,} price observations from {n_stations:,} stations  \n\n"
            f"During **{start_month} to {end_month}** for **{fuel_type}**, "
            f"the average price between **18–21h** is **{evening_ct:.2f} ct/L above the daily minimum**, "
            f"and between **6–9h** it is **{morning_ct:.2f} ct/L**. "
            f"Evening refueling is therefore on average **{diff_ct:.2f} ct/L** cheaper.  \n\n"
            f"The best single hour is **{best_hour}:00h**. "
            f"At this hour the price is on average only **{best_hour_ct:.2f} ct/L above the daily minimum**, "
            f"and the probability of hitting the exact daily minimum is **{best_prob:.1f} %**.  \n\n"
            f"The best contiguous window is **{best_window['start']}:00–{best_window['end']}:59h** "
            f"with average **{best_window_ct:.2f} ct/L above the daily minimum**."
        )

    def make_hour_plot(hour_stats: pl.DataFrame, stat: str, title_suffix: str) -> go.Figure:
        value_col = "mean_diff" if stat == "mean" else "median_diff"

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=hour_stats["hour"].to_list(),
                y=(hour_stats[value_col] * 100).to_list(),
                mode="lines+markers",
                name="ct/L above daily minimum",
            )
        )
        fig.update_layout(
            title=f"Average distance to daily minimum by hour{title_suffix}",
            xaxis_title="Hour",
            yaxis_title="ct/L above daily minimum",
            height=450,
        )
        return fig

    def make_heatmap(heat_stats: pl.DataFrame, stat: str, title_suffix: str) -> go.Figure:
        value_col = "mean_diff" if stat == "mean" else "median_diff"

        heat = (
            heat_stats.pivot(
                values=value_col,
                index="weekday",
                on="hour",
                aggregate_function="first",
            )
            .sort("weekday")
        )

        y = [weekday_labels[int(v)] for v in heat["weekday"].to_list()]
        x = [c for c in heat.columns if c != "weekday"]
        z = heat.drop("weekday").to_numpy() * 100

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=x,
                y=y,
                colorbar=dict(title="ct/L above daily minimum"),
            )
        )
        fig.update_layout(
            title=f"Heatmap: Weekday × Hour{title_suffix}",
            xaxis_title="Hour",
            yaxis_title="Weekday",
            height=500,
        )
        return fig

    def render(change=None):
        with output:
            clear_output(wait=True)

            start_month, end_month = month_slider.value
            fuel_type = fuel_dropdown.value
            stat = stat_dropdown.value
            city = city_text.value
            brand = brand_text.value
            plz_prefix = plz_text.value

            lf = load_period_lazy(start_month, end_month)
            if lf is None:
                print("No data found.")
                return

            lf = apply_filters_lazy(lf, fuel_type, city, brand, plz_prefix)

            meta = compute_meta_stats(lf)
            if meta["n_obs"] == 0:
                print("No data after applying filters.")
                return

            hour_stats = compute_hour_stats(lf)
            heat_stats = compute_heatmap_stats(lf)

            filter_parts = [fuel_type]
            if city.strip():
                filter_parts.append(f"City~{city.strip()}")
            if brand.strip():
                filter_parts.append(f"Brand~{brand.strip()}")
            if plz_prefix.strip():
                filter_parts.append(f"PLZ~{plz_prefix.strip()}")

            title_suffix = " | " + ", ".join(filter_parts)

            fig1 = make_hour_plot(hour_stats, stat, title_suffix)
            fig1.show()

            fig2 = make_heatmap(heat_stats, stat, title_suffix)
            fig2.show()

            summary = build_summary_text(
                hour_stats=hour_stats,
                meta=meta,
                start_month=start_month,
                end_month=end_month,
                fuel_type=fuel_type,
                city=city,
                brand=brand,
                plz_prefix=plz_prefix,
            )
            display(Markdown(summary))

    month_slider.observe(render, names="value")
    fuel_dropdown.observe(render, names="value")
    stat_dropdown.observe(render, names="value")
    city_text.observe(render, names="value")
    brand_text.observe(render, names="value")
    plz_text.observe(render, names="value")

    controls = widgets.VBox([
        month_slider,
        widgets.HBox([fuel_dropdown, stat_dropdown]),
        widgets.HBox([city_text, brand_text, plz_text]),
    ])

    display(controls, output)
    render()