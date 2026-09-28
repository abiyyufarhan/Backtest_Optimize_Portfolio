"""Streamlit dashboard for precomputed backtest and portfolio artifacts.

This app only reads files from ``data/processed``. It does not run a backtest,
portfolio optimization, model fitting, or other expensive calculation. The
upstream notebooks/scripts should export the following optional artifacts:

* ``backtest_equity.csv`` / ``equity_curves.csv`` with date and equity columns;
* ``backtest_metrics.csv`` / ``evaluation_metrics_all.csv``;
* ``backtest_weights.csv`` / ``portfolio_weights.csv``;
* ``cleaning_log.csv`` for the Data Quality tab.

The loader also accepts long-form files with ``strategy``, ``equity``,
``ticker``, and ``weight`` columns and wide-form equity files.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def _first_existing(names: Iterable[str]) -> Optional[Path]:
    for name in names:
        path = PROCESSED / name
        if path.exists():
            return path
    return None


def _date_column(frame: pd.DataFrame) -> Optional[str]:
    for column in ("date", "tanggal", "datetime", "timestamp"):
        if column in frame.columns:
            return column
    return None


def _name_column(frame: pd.DataFrame) -> Optional[str]:
    for column in ("strategy", "portfolio", "name", "model"):
        if column in frame.columns:
            return column
    return None


def _parse_cost_multiple(value: object) -> float:
    match = re.search(r"(?:cost|biaya)[^0-9]*([0-9]+(?:\.[0-9]+)?)x", str(value).lower())
    return float(match.group(1)) if match else np.nan


@st.cache_data(show_spinner=False)
def load_equity() -> tuple[pd.DataFrame, Optional[Path]]:
    """Load equity curves into long form: date, name, equity, cost_multiple."""
    path = _first_existing(
        [
            "backtest_equity.csv",
            "equity_curves.csv",
            "equity_curve.csv",
            "evaluation_equity.csv",
            "portfolio_optimization_equity.csv",
        ]
    )
    if path is None:
        return pd.DataFrame(columns=["date", "name", "equity", "cost_multiple"]), None

    raw = pd.read_csv(path)
    date_col = _date_column(raw)
    if date_col is None:
        raw = raw.rename(columns={raw.columns[0]: "date"})
        date_col = "date"
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=[date_col])

    name_col = _name_column(raw)
    value_candidates = ["equity", "equity_value", "nav", "value"]
    value_col = next((column for column in value_candidates if column in raw.columns), None)
    cost_col = next((column for column in ("cost_multiple", "cost", "biaya_multiple") if column in raw.columns), None)

    if name_col and value_col:
        long = raw.rename(columns={date_col: "date", name_col: "name", value_col: "equity"})[
            ["date", "name", "equity"]
        ].copy()
        long["cost_multiple"] = pd.to_numeric(raw[cost_col], errors="coerce") if cost_col else np.nan
    else:
        value_columns = [column for column in raw.columns if column != date_col]
        long = raw.rename(columns={date_col: "date"}).melt(
            id_vars="date", value_vars=value_columns, var_name="name", value_name="equity"
        )
        long["cost_multiple"] = long["name"].map(_parse_cost_multiple)

    long["equity"] = pd.to_numeric(long["equity"], errors="coerce")
    long["date"] = pd.to_datetime(long["date"], errors="coerce")
    long = long.dropna(subset=["date", "equity"]).sort_values(["name", "date"]).reset_index(drop=True)
    return long, path


@st.cache_data(show_spinner=False)
def load_metrics() -> tuple[pd.DataFrame, Optional[Path]]:
    """Load precomputed metrics without recalculating them in the app."""
    path = _first_existing(
        [
            "backtest_metrics.csv",
            "metrics.csv",
            "evaluation_metrics_all.csv",
            "portfolio_optimization_performance.csv",
        ]
    )
    if path is None:
        return pd.DataFrame(), None
    raw = pd.read_csv(path)
    if "Unnamed: 0" in raw.columns and not _name_column(raw):
        raw = raw.rename(columns={"Unnamed: 0": "name"})
    if "portfolio" in raw.columns and "name" not in raw.columns:
        raw = raw.rename(columns={"portfolio": "name"})
    if "name" not in raw.columns:
        raw = raw.rename(columns={raw.columns[0]: "name"})
    if "cost_multiple" not in raw.columns:
        raw["cost_multiple"] = np.nan
    return raw, path


def _read_multiindex_weights(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=[0, 1], index_col=0)
    raw.index = pd.to_datetime(raw.index, errors="coerce")
    raw = raw.loc[raw.index.notna()]
    rows = []
    for method in raw.columns.get_level_values(0).unique():
        subset = raw[method]
        for date, row in subset.iterrows():
            for ticker, weight in row.items():
                if str(ticker).endswith(".JK"):
                    rows.append({"date": date, "name": method, "ticker": ticker, "weight": weight})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_weights() -> tuple[pd.DataFrame, Optional[Path]]:
    """Load precomputed weights into long form: date, name, ticker, weight."""
    path = _first_existing(
        [
            "backtest_weights.csv",
            "weights.csv",
            "portfolio_weights.csv",
            "portfolio_optimization_weights.csv",
        ]
    )
    if path is None:
        return pd.DataFrame(columns=["date", "name", "ticker", "weight"]), None

    try:
        raw = pd.read_csv(path)
    except (pd.errors.ParserError, ValueError):
        return _read_multiindex_weights(path), path

    date_col = _date_column(raw)
    name_col = _name_column(raw)
    ticker_col = next((column for column in ("ticker", "stock", "symbol") if column in raw.columns), None)
    weight_col = next((column for column in ("weight", "target_weight", "value") if column in raw.columns), None)
    if date_col and name_col and ticker_col and weight_col:
        out = raw.rename(
            columns={date_col: "date", name_col: "name", ticker_col: "ticker", weight_col: "weight"}
        )[["date", "name", "ticker", "weight"]].copy()
    elif date_col and name_col:
        tickers = [column for column in raw.columns if str(column).endswith(".JK")]
        out = raw.melt(id_vars=[date_col, name_col], value_vars=tickers, var_name="ticker", value_name="weight")
        out = out.rename(columns={date_col: "date", name_col: "name"})
    else:
        # The portfolio optimizer exports a two-level column CSV.
        out = _read_multiindex_weights(path)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    out = out.dropna(subset=["date", "weight"]).sort_values(["name", "date", "ticker"])
    return out.reset_index(drop=True), path


@st.cache_data(show_spinner=False)
def load_quality_log() -> tuple[pd.DataFrame, Optional[Path]]:
    path = _first_existing(["data_quality_log.csv", "cleaning_log.csv"])
    if path is None:
        return pd.DataFrame(), None
    return pd.read_csv(path), path


def _select_equity(equity: pd.DataFrame, name: str, cost_multiple: float) -> pd.DataFrame:
    selected = equity.loc[equity["name"].eq(name)].copy()
    available_costs = selected["cost_multiple"].dropna().unique()
    if len(available_costs):
        selected = selected.loc[np.isclose(selected["cost_multiple"], cost_multiple)]
    return selected.sort_values("date")


def _weight_name_aliases(name: Optional[str]) -> set[str]:
    """Accept portfolio names with or without the dashboard's prefix."""
    if not name:
        return set()
    clean = str(name)
    bare = clean.removeprefix("Portfolio_")
    return {clean, bare, f"Portfolio_{bare}"}


def _normalize_series(frame: pd.DataFrame) -> pd.Series:
    series = frame.set_index("date")["equity"].sort_index()
    return series / series.iloc[0]


def _drawdown(series: pd.Series) -> pd.Series:
    return series / series.cummax() - 1.0


def _monthly_heatmap(series: pd.Series) -> pd.DataFrame:
    monthly = series.resample("ME").last().pct_change().dropna()
    if monthly.empty:
        return pd.DataFrame()
    table = monthly.to_frame("return")
    table["year"] = table.index.year
    table["month"] = table.index.month
    return table.pivot(index="year", columns="month", values="return").reindex(columns=range(1, 13))


def _format_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    percent_columns = [column for column in ("CAGR", "volatility_annual", "Max Drawdown", "win_rate") if column in metrics]
    result = metrics.copy()
    for column in percent_columns:
        result[column] = result[column].map(lambda value: f"{value:.2%}" if pd.notna(value) else "—")
    for column in ("Sharpe", "Sortino", "Calmar", "turnover_annual"):
        if column in result:
            result[column] = result[column].map(lambda value: f"{value:.2f}" if pd.notna(value) else "—")
    return result


def main() -> None:
    st.set_page_config(page_title="Quant Finance Dashboard", layout="wide")
    st.title("Quant Finance Strategy Dashboard")
    st.caption("Dashboard read-only: semua equity curve, metrics, dan bobot dibaca dari data/processed.")

    equity, equity_path = load_equity()
    metrics, metrics_path = load_metrics()
    weights, weights_path = load_weights()
    quality_log, quality_path = load_quality_log()

    if equity.empty:
        st.error("Artefak equity curve belum ditemukan di data/processed.")
        st.info(
            "Simpan hasil backtest sebagai backtest_equity.csv atau equity_curves.csv "
            "dengan kolom date, name/strategy, equity, dan opsional cost_multiple."
        )
    elif equity_path:
        st.sidebar.caption(f"Equity source: {equity_path.name}")

    strategy_options = sorted(equity["name"].dropna().unique().tolist()) if not equity.empty else []
    benchmark_names = {"benchmark_lq45", "benchmark_equal_weight", "lq45", "LQ45"}
    strategy_options = [name for name in strategy_options if name not in benchmark_names]
    selected_strategy = st.sidebar.selectbox("Strategi / portofolio", strategy_options) if strategy_options else None

    cost_options = [0.0, 1.0, 2.0]
    selected_cost = st.sidebar.selectbox("Biaya transaksi", cost_options, index=1, format_func=lambda value: f"{value:.0f}x")
    selected_stocks = st.sidebar.multiselect(
        "Filter saham pada tabel bobot",
        sorted(weights["ticker"].dropna().unique().tolist()) if not weights.empty else [],
        help="Pilihan ini hanya memfilter baris yang ditampilkan; tidak mengubah bobot atau menghitung ulang portfolio.",
    )

    if equity.empty:
        date_min = date_max = pd.Timestamp("2020-01-01")
    else:
        date_min = equity["date"].min()
        date_max = equity["date"].max()
    selected_period = st.sidebar.date_input(
        "Periode", value=(date_min.date(), date_max.date()), min_value=date_min.date(), max_value=date_max.date()
    )
    if isinstance(selected_period, tuple) and len(selected_period) == 2:
        period_start, period_end = map(pd.Timestamp, selected_period)
    else:
        period_start, period_end = date_min, date_max

    tab_overview, tab_weights, tab_quality = st.tabs(["Overview", "Portfolio Weights", "Data Quality & Limitations"])
    with tab_overview:
        if selected_strategy is None:
            st.warning("Belum ada equity curve yang dapat ditampilkan.")
        else:
            selected = _select_equity(equity, selected_strategy, selected_cost)
            selected = selected.loc[selected["date"].between(period_start, period_end)]
            if selected.empty:
                st.warning("Tidak ada data untuk strategi, biaya, dan periode yang dipilih.")
            else:
                selected_series = _normalize_series(selected)
                plot_frame = selected_series.rename(selected_strategy).to_frame()
                benchmark_candidates = [name for name in equity["name"].unique() if str(name).lower() in {"benchmark_lq45", "lq45"}]
                benchmark_name = benchmark_candidates[0] if benchmark_candidates else None
                if benchmark_name:
                    benchmark = _select_equity(equity, benchmark_name, selected_cost)
                    benchmark = benchmark.loc[benchmark["date"].between(period_start, period_end)]
                    if not benchmark.empty:
                        plot_frame["LQ45"] = _normalize_series(benchmark).reindex(plot_frame.index).ffill()

                st.subheader("Equity Curve vs LQ45")
                st.line_chart(plot_frame)

                st.subheader("Drawdown")
                drawdown_frame = pd.DataFrame({selected_strategy: _drawdown(selected_series)})
                if "LQ45" in plot_frame:
                    drawdown_frame["LQ45"] = _drawdown(plot_frame["LQ45"])
                st.line_chart(drawdown_frame)

                st.subheader("Metrik")
                if metrics.empty:
                    st.info("File metrics belum ditemukan di data/processed.")
                else:
                    metric_view = metrics.copy()
                    if "cost_multiple" in metric_view.columns and metric_view["cost_multiple"].notna().any():
                        metric_view = metric_view.loc[
                            metric_view["cost_multiple"].isna() | np.isclose(metric_view["cost_multiple"], selected_cost)
                        ]
                    metric_view = metric_view.loc[metric_view["name"].isin([selected_strategy, benchmark_name, "benchmark_equal_weight"])]
                    st.dataframe(_format_metrics(metric_view), use_container_width=True, hide_index=True)

                st.subheader("Heatmap Return Bulanan")
                heatmap = _monthly_heatmap(selected_series)
                if heatmap.empty:
                    st.info("Belum cukup data untuk heatmap bulanan.")
                else:
                    heatmap.columns = [pd.Timestamp(2000, month, 1).strftime("%b") for month in heatmap.columns]
                    fig_heatmap = go.Figure(
                        data=go.Heatmap(
                            z=heatmap.to_numpy() * 100,
                            x=heatmap.columns,
                            y=heatmap.index.astype(str),
                            colorscale="RdYlGn",
                            colorbar=dict(title="Return %"),
                            hovertemplate="%{y} %{x}: %{z:.2f}%<extra></extra>",
                        )
                    )
                    fig_heatmap.update_layout(height=420, margin=dict(l=20, r=20, t=20, b=20))
                    st.plotly_chart(fig_heatmap, use_container_width=True)

    with tab_weights:
        st.subheader("Bobot Portofolio")
        if weights.empty:
            st.info("File bobot belum ditemukan di data/processed.")
        else:
            weight_aliases = _weight_name_aliases(selected_strategy)
            weight_view = weights.loc[weights["name"].isin(weight_aliases)].copy() if selected_strategy else weights.iloc[0:0]
            if selected_strategy and not weights["name"].isin(weight_aliases).any():
                if str(selected_strategy).startswith("ML_"):
                    st.info(
                        "ML signal adalah strategi market-timing berbasis arah LQ45. "
                        "Artefak bobot saham per-ticker belum disimpan untuk strategi ini; "
                        "pilihan saham tidak mengubah posisi ML secara interaktif."
                    )
                else:
                    st.info(
                        "Bobot untuk strategi ini belum tersedia di portfolio_optimization_weights.csv. "
                        "File tersebut hanya memuat bobot portfolio yang diekspor upstream."
                    )
            weight_view = weight_view.loc[weight_view["date"].between(period_start, period_end)]
            if selected_stocks:
                weight_view = weight_view.loc[weight_view["ticker"].isin(selected_stocks)]
            has_strategy_weights = bool(selected_strategy) and weights["name"].isin(weight_aliases).any()
            if weight_view.empty and has_strategy_weights:
                st.info("Tidak ada bobot untuk strategi dan periode yang dipilih.")
            elif not weight_view.empty:
                st.dataframe(
                    weight_view.pivot_table(index="date", columns="ticker", values="weight", aggfunc="last")
                    .style.format("{:.2%}"),
                    use_container_width=True,
                )

    with tab_quality:
        st.subheader("Data Quality Log")
        if quality_log.empty:
            st.warning("cleaning_log.csv belum ditemukan di data/processed.")
        else:
            st.dataframe(quality_log, use_container_width=True, hide_index=True)
            if quality_path:
                st.caption(f"Source: {quality_path}")

        st.subheader("Limitations")
        st.markdown(
            """
            - Dashboard tidak menghitung ulang backtest, optimasi portofolio, atau model ML; hasil harus dibuat terlebih dahulu oleh script/notebook upstream.
            - Biaya transaksi pada dashboard hanya sebaik artefak precomputed yang dibaca. Pastikan fee beli, fee jual, dan slippage sudah sesuai broker.
            - Release lag makro adalah pendekatan konfigurasi, bukan kalender publikasi aktual; revisi data dan vintage belum dimodelkan penuh.
            - Ranking kualitas saham adalah snapshot 2026 dan hanya filter awal, bukan sinyal historis.
            - Performa historis tidak menjamin hasil masa depan; hasil dapat berubah bila universe, kalender trading, atau data mentah berubah.
            """
        )


if __name__ == "__main__":
    main()
