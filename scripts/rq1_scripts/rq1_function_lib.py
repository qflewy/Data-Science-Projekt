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
    data_dir: str | Path = r"D:/data/derived/station_price_observations",
    file_pattern: str = "station_price_observations_*.parquet",
):
    data_dir = Path(data_dir)

    # ------------------------------------------------------------
    # 1) Monatsdateien finden
    # ------------------------------------------------------------
    files = sorted(data_dir.glob(file_pattern))
    month_pattern = re.compile(r"(\d{4})_(\d{2})")

    month_files: dict[str, Path] = {}
    for f in files:
        m = month_pattern.search(f.stem)
        if m:
            month_key = f"{m.group(1)}-{m.group(2)}"
            month_files[month_key] = f

    if not month_files:
        raise FileNotFoundError(f"Keine passenden Dateien in {data_dir} gefunden.")

    month_keys = sorted(month_files.keys())

    # ------------------------------------------------------------
    # 2) Widgets
    # ------------------------------------------------------------
    month_slider = widgets.SelectionRangeSlider(
        options=month_keys,
        index=(0, len(month_keys) - 1),
        description="Zeitraum",
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
        description="Maß",
    )

    city_text = widgets.Text(
        value="",
        description="City",
        placeholder="z. B. Hamburg",
    )

    brand_text = widgets.Text(
        value="",
        description="Brand",
        placeholder="z. B. ARAL",
    )

    plz_text = widgets.Text(
        value="",
        description="PLZ",
        placeholder="z. B. 24",
    )

    output = widgets.Output()

    weekday_labels = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]

    # ------------------------------------------------------------
    # 3) Zeitraum laden
    # ------------------------------------------------------------
    def load_period(start_month: str, end_month: str) -> pl.DataFrame:
        selected_months = [m for m in month_keys if start_month <= m <= end_month]
        selected_files = [month_files[m] for m in selected_months]

        if not selected_files:
            return pl.DataFrame()

        lf = pl.concat([pl.scan_parquet(f) for f in selected_files])

        return lf.collect(engine="streaming")

    # ------------------------------------------------------------
    # 4) Filter anwenden
    # ------------------------------------------------------------
    def apply_filters(
        df: pl.DataFrame,
        fuel_type: str,
        city: str,
        brand: str,
        plz_prefix: str,
    ) -> pl.DataFrame:
        out = df.filter(pl.col("fuel_type") == fuel_type)

        if city.strip():
            out = out.filter(
                pl.col("city")
                .cast(pl.Utf8)
                .str.to_lowercase()
                .str.contains(city.strip().lower(), literal=True)
            )

        if brand.strip():
            out = out.filter(
                pl.col("brand")
                .fill_null("")
                .cast(pl.Utf8)
                .str.to_lowercase()
                .str.contains(brand.strip().lower(), literal=True)
            )

        if plz_prefix.strip():
            out = out.filter(
                pl.col("post_code")
                .cast(pl.Utf8)
                .str.starts_with(plz_prefix.strip())
            )

        return out

    # ------------------------------------------------------------
    # 5) Analyse vorbereiten
    # ------------------------------------------------------------
    def prepare_analysis(df: pl.DataFrame) -> pl.DataFrame:
        # Tagesminimum je Tankstelle, Kraftstoff, Tag
        df = df.with_columns(
            pl.col("price")
            .min()
            .over(["station_id", "fuel_type", "date"])
            .alias("daily_min")
        )

        # Abstand zum Tagesminimum
        df = df.with_columns(
            (pl.col("price") - pl.col("daily_min")).alias("diff_to_min")
        )

        # Exakt Tagesminimum?
        df = df.with_columns(
            (pl.col("price") == pl.col("daily_min")).cast(pl.Float64).alias("is_daily_min")
        )

        return df

    # ------------------------------------------------------------
    # 6) Text erzeugen
    # ------------------------------------------------------------
    def build_summary_text(
        df: pl.DataFrame,
        start_month: str,
        end_month: str,
        fuel_type: str,
        city: str,
        brand: str,
        plz_prefix: str,
    ) -> str:
        hour_stats = (
            df.group_by("hour")
            .agg([
                pl.mean("diff_to_min").alias("mean_diff"),
                pl.median("diff_to_min").alias("median_diff"),
                pl.mean("is_daily_min").alias("prob_min"),
            ])
            .sort("hour")
        )

        best_hour_row = hour_stats.sort("mean_diff").row(0, named=True)
        best_hour = int(best_hour_row["hour"])
        best_hour_ct = float(best_hour_row["mean_diff"]) * 100
        best_prob = float(best_hour_row["prob_min"]) * 100

        evening = df.filter(pl.col("hour").is_between(18, 21, closed="both"))
        morning = df.filter(pl.col("hour").is_between(6, 9, closed="both"))

        evening_ct = float(evening.select(pl.mean("diff_to_min")).item()) * 100
        morning_ct = float(morning.select(pl.mean("diff_to_min")).item()) * 100
        diff_ct = morning_ct - evening_ct

        # Bestes 4-Stunden-Abendfenster optional sauberer bestimmen
        # Hier zusätzlich: bestes 2h / 3h / 4h Fenster aus allen Stunden
        window_results = []
        hour_map = {row["hour"]: row["mean_diff"] for row in hour_stats.to_dicts()}

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

        # Filterbeschreibung
        filters = [fuel_type]
        if city.strip():
            filters.append(f"Stadt enthält '{city.strip()}'")
        if brand.strip():
            filters.append(f"Marke enthält '{brand.strip()}'")
        if plz_prefix.strip():
            filters.append(f"PLZ beginnt mit '{plz_prefix.strip()}'")

        filter_text = ", ".join(filters)

        # Zusatzinfos zu Beobachtungen
        n_obs = df.height
        n_stations = df.select(pl.col("station_id").n_unique()).item()

        return (
            f"**Zeitraum:** {start_month} bis {end_month}  \n"
            f"**Filter:** {filter_text}  \n"
            f"**Beobachtungen:** {n_obs:,} Preisbeobachtungen aus {n_stations:,} Tankstellen  \n\n"
            f"Im Zeitraum **{start_month} bis {end_month}** liegen die Preise für **{fuel_type}** "
            f"zwischen **18–21 Uhr** im Schnitt **{evening_ct:.2f} ct/L über dem Tagesminimum**, "
            f"morgens zwischen **6–9 Uhr** dagegen **{morning_ct:.2f} ct/L**. "
            f"Das Abendtanken ist damit im Mittel um **{diff_ct:.2f} ct/L** günstiger.  \n\n"
            f"Die statistisch beste Einzelstunde ist **{best_hour}:00 Uhr**. "
            f"Zu dieser Stunde liegt der Preis im Schnitt nur **{best_hour_ct:.2f} ct/L über dem Tagesminimum**, "
            f"und die Wahrscheinlichkeit, genau das Tagesminimum zu treffen, beträgt **{best_prob:.1f} %**.  \n\n"
            f"Das beste zusammenhängende Zeitfenster ist **{best_window['start']}:00–{best_window['end']}:59 Uhr** "
            f"mit durchschnittlich **{best_window_ct:.2f} ct/L über dem Tagesminimum**."
        )

    # ------------------------------------------------------------
    # 7) Plots
    # ------------------------------------------------------------
    def make_hour_plot(df: pl.DataFrame, stat: str, title_suffix: str) -> go.Figure:
        hour_stats = (
            df.group_by("hour")
            .agg([
                pl.mean("diff_to_min").alias("mean_diff"),
                pl.median("diff_to_min").alias("median_diff"),
                pl.mean("is_daily_min").alias("prob_min"),
            ])
            .sort("hour")
        )

        value_col = "mean_diff" if stat == "mean" else "median_diff"

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=hour_stats["hour"].to_list(),
                y=(hour_stats[value_col] * 100).to_list(),
                mode="lines+markers",
                name="ct/L über Tagesminimum",
            )
        )
        fig.update_layout(
            title=f"Durchschnittliche Preisnähe zum Tagesminimum nach Stunde{title_suffix}",
            xaxis_title="Stunde",
            yaxis_title="ct/L über Tagesminimum",
            height=450,
        )
        return fig

    def make_heatmap(df: pl.DataFrame, stat: str, title_suffix: str) -> go.Figure:
        agg = (
            df.group_by(["weekday", "hour"])
            .agg([
                pl.mean("diff_to_min").alias("mean_diff"),
                pl.median("diff_to_min").alias("median_diff"),
            ])
            .sort(["weekday", "hour"])
        )

        value_col = "mean_diff" if stat == "mean" else "median_diff"

        heat = (
            agg.pivot(
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
                colorbar=dict(title="ct/L über Tagesminimum"),
            )
        )
        fig.update_layout(
            title=f"Heatmap: Wochentag × Stunde{title_suffix}",
            xaxis_title="Stunde",
            yaxis_title="Wochentag",
            height=500,
        )
        return fig

    # ------------------------------------------------------------
    # 8) Render
    # ------------------------------------------------------------
    def render(change=None):
        with output:
            clear_output(wait=True)

            start_month, end_month = month_slider.value
            fuel_type = fuel_dropdown.value
            stat = stat_dropdown.value
            city = city_text.value
            brand = brand_text.value
            plz_prefix = plz_text.value

            df = load_period(start_month, end_month)

            if df.is_empty():
                print("Keine Daten gefunden.")
                return

            df = apply_filters(df, fuel_type, city, brand, plz_prefix)

            if df.is_empty():
                print("Keine Daten nach Anwendung der Filter.")
                return

            df = prepare_analysis(df)

            filter_parts = [fuel_type]
            if city.strip():
                filter_parts.append(f"City~{city.strip()}")
            if brand.strip():
                filter_parts.append(f"Brand~{brand.strip()}")
            if plz_prefix.strip():
                filter_parts.append(f"PLZ~{plz_prefix.strip()}")

            title_suffix = " | " + ", ".join(filter_parts)

            fig1 = make_hour_plot(df, stat, title_suffix)
            fig1.show()

            fig2 = make_heatmap(df, stat, title_suffix)
            fig2.show()

            summary = build_summary_text(
                df=df,
                start_month=start_month,
                end_month=end_month,
                fuel_type=fuel_type,
                city=city,
                brand=brand,
                plz_prefix=plz_prefix,
            )
            display(Markdown(summary))

    # Widget-Events
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