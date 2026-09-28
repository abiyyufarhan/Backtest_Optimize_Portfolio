"""Cleaning functions for the quant-finance raw datasets.

The functions in this module never modify their input DataFrames in place.
They return clean, date-indexed data and optionally append auditable entries
to a shared cleaning log.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, MutableSequence, Optional

import numpy as np
import pandas as pd


TARGET_PRICE_FILL_DATES = ("2019-06-19", "2026-08-25")


def _record(log: Optional[MutableSequence[dict[str, Any]]], dataset: str,
            action: str, detail: str, affected_rows: int = 0) -> None:
    if log is not None:
        log.append({
            "dataset": dataset,
            "action": action,
            "detail": detail,
            "affected_rows": int(affected_rows),
        })


def _rename_date_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    date_candidates = [c for c in out.columns if str(c).strip().lower() in {"date", "tanggal"}]
    if not date_candidates:
        raise KeyError("Kolom tanggal tidak ditemukan.")
    date_col = date_candidates[0]
    out = out.rename(columns={date_col: "date"})
    raw_dates = out["date"].astype(str).str.strip()
    is_iso = raw_dates.str.match(r"^\d{4}-\d{2}-\d{2}$").all()
    out["date"] = pd.to_datetime(out["date"], errors="raise", dayfirst=(date_col.lower() == "tanggal" and not is_iso))
    return out


def _assert_dates(out: pd.DataFrame) -> None:
    assert "date" in out.columns
    assert not out["date"].duplicated().any(), "Tanggal duplikat ditemukan."
    assert out["date"].is_monotonic_increasing, "Tanggal tidak monoton naik."


def _assert_tickers(out: pd.DataFrame) -> None:
    assert "Ticker" in out.columns
    assert not out["Ticker"].duplicated().any(), "Ticker duplikat ditemukan."


def _parse_indonesian_number(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (series.astype(str)
               .str.replace(r"[^0-9,.-]", "", regex=True)
               .str.replace(".", "", regex=False)
               .str.replace(",", ".", regex=False))
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_thousands_number(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(r"[^0-9-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _quarter_end(values: Iterable[Any]) -> pd.Series:
    """Convert labels such as 2016 Q1 and 2016-Q1 to quarter-end dates."""
    values = pd.Series(values)
    text = values.astype(str).str.strip().str.replace("-", " ", regex=False)
    match = text.str.extract(r"^(?P<year>\d{4})\s+Q(?P<quarter>[1-4])$")
    if match["year"].notna().all():
        periods = pd.PeriodIndex(
            [pd.Period(year=int(year), quarter=int(quarter), freq="Q")
             for year, quarter in match.itertuples(index=False, name=None)],
            freq="Q",
        )
        return pd.Series(periods.to_timestamp(how="end").normalize(), index=values.index)
    parsed = pd.to_datetime(values, errors="raise", dayfirst=True)
    return pd.Series(parsed, index=values.index)


def clean_prices(
    adj_close: pd.DataFrame,
    index_lq45: pd.DataFrame,
    log: Optional[MutableSequence[dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Align adjusted prices to the LQ45 trading calendar and repair target NaNs."""
    adj = _rename_date_column(adj_close)
    idx = _rename_date_column(index_lq45)
    idx = idx.sort_values("date")
    calendar = pd.DatetimeIndex(idx["date"])
    assert not calendar.duplicated().any(), "Kalender LQ45 memiliki tanggal duplikat."
    assert calendar.is_monotonic_increasing, "Kalender LQ45 tidak monoton naik."

    adj = adj.sort_values("date").set_index("date")
    asset_cols = [c for c in adj.columns if c not in {"Date", "date"}]
    target_dates = pd.to_datetime(list(TARGET_PRICE_FILL_DATES))
    filled = adj[asset_cols].ffill(limit=1)
    for target in target_dates:
        if target not in adj.index:
            raise AssertionError(f"Tanggal target {target.date()} tidak ada di adj_close.")
        before = int(adj.loc[target, asset_cols].isna().sum())
        if before:
            adj.loc[target, asset_cols] = filled.loc[target, asset_cols]
            after = int(adj.loc[target, asset_cols].isna().sum())
            assert after == 0, f"Forward-fill gagal pada {target.date()}."
            _record(log, "prices", "forward_fill", f"{target.date()}; limit=1", before)

    holiday_dates = pd.DatetimeIndex(adj.index).difference(calendar)
    _record(log, "prices", "drop_exchange_holidays",
            f"Dikeluarkan dari kalender resmi: {len(holiday_dates)} tanggal.", len(holiday_dates))

    out = adj.reindex(calendar)
    if len(idx) and float(idx.iloc[-1]["Volume"]) == 0:
        last_date = calendar[-1]
        out = out.drop(index=last_date)
        _record(log, "prices", "drop_zero_volume_tail",
                f"Baris terakhir {last_date.date()} dibuang karena Volume index_lq45 = 0.", 1)

    out.index.name = "date"
    out = out.reset_index()
    _assert_dates(out)
    expected_len = len(calendar) - 1 if float(idx.iloc[-1]["Volume"]) == 0 else len(calendar)
    assert len(out) == expected_len
    return out


def clean_index_lq45(
    index_lq45: pd.DataFrame,
    trading_calendar: Optional[pd.DatetimeIndex] = None,
    log: Optional[MutableSequence[dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Reindex the LQ45 index to the supplied official trading calendar."""
    out = _rename_date_column(index_lq45).sort_values("date")
    if trading_calendar is None:
        trading_calendar = pd.DatetimeIndex(out["date"])
    assert not trading_calendar.duplicated().any()
    assert trading_calendar.is_monotonic_increasing
    out = out.set_index("date").reindex(trading_calendar)
    out.index.name = "date"
    out = out.reset_index()
    out = out.rename(columns={
        "Adj Close": "adj_close", "Close": "close", "High": "high",
        "Low": "low", "Open": "open", "Volume": "volume",
    })
    _assert_dates(out)
    assert not out["date"].duplicated().any()
    return out


def clean_usdidr(
    usdidr: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
    log: Optional[MutableSequence[dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Align USD/IDR observations to the official calendar and ffill up to 3 days."""
    out = _rename_date_column(usdidr).sort_values("date")
    assert not trading_calendar.duplicated().any()
    assert trading_calendar.is_monotonic_increasing
    out = out.set_index("date").reindex(trading_calendar)
    value_cols = [c for c in out.columns]
    missing_before = int(out[value_cols].isna().sum().sum())
    out[value_cols] = out[value_cols].ffill(limit=3)
    filled_cells = missing_before - int(out[value_cols].isna().sum().sum())
    if filled_cells:
        _record(log, "usdidr", "forward_fill", "Reindex kalender resmi; limit=3 hari.", filled_cells)
    out.index.name = "date"
    out = out.reset_index()
    out = out.rename(columns={
        "Adj Close": "adj_close", "Close": "close", "High": "high",
        "Low": "low", "Open": "open", "Volume": "volume",
    })
    _assert_dates(out)
    assert not out["date"].duplicated().any()
    return out


def clean_pdb(
    pdb: pd.DataFrame,
    correction_path: Optional[Path] = None,
    log: Optional[MutableSequence[dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Parse Indonesian PDB values, convert quarters to quarter-end dates, and apply corrections."""
    raw = pdb.copy()
    period_col = "Periode"
    raw["date"] = _quarter_end(raw[period_col])
    out = pd.DataFrame({
        "date": raw["date"],
        "pdb_adhb_triliun_rp": _parse_indonesian_number(raw["PDB ADHB (Triliun Rp)"]),
        "pdb_adhk_2010_triliun_rp": _parse_indonesian_number(raw["PDB ADHK 2010 (Triliun Rp)"]),
        "pertumbuhan_yoy_pct": _parse_indonesian_number(raw["Pertumbuhan y-on-y (%)"]),
    })

    applied = 0
    if correction_path is not None and Path(correction_path).exists():
        correction = pd.read_csv(correction_path)
        correction_period = "Periode" if "Periode" in correction.columns else "date"
        correction["date"] = (_quarter_end(correction[correction_period])
                               if correction_period != "date" else pd.to_datetime(correction["date"], errors="raise"))
        aliases = {
            "PDB ADHB (Triliun Rp)": "pdb_adhb_triliun_rp",
            "PDB ADHK 2010 (Triliun Rp)": "pdb_adhk_2010_triliun_rp",
            "Pertumbuhan y-on-y (%)": "pertumbuhan_yoy_pct",
        }
        correction = correction.rename(columns=aliases)
        for col in [c for c in out.columns if c != "date"]:
            if col in correction.columns:
                values = _parse_indonesian_number(correction[col])
                if col == "pdb_adhk_2010_triliun_rp" and values.dropna().median() > 10000:
                    values = values / 1000
                    _record(log, "pdb", "unit_conversion",
                            "Nilai ADHK koreksi manual dikonversi dari miliar Rp ke triliun Rp.",
                            int(values.notna().sum()))
                updates = pd.DataFrame({"date": correction["date"], col: values}).dropna(subset=[col])
                out = out.set_index("date")
                for date, value in updates.set_index("date")[col].items():
                    if date in out.index:
                        out.loc[date, col] = value
                        applied += 1
                out = out.reset_index()
        _record(log, "pdb", "manual_correction", f"Menggunakan {Path(correction_path).name}.", applied)

    out = out.sort_values("date").reset_index(drop=True)
    _assert_dates(out)
    assert not out["date"].duplicated().any()
    return out


def clean_inflation(
    inflation: pd.DataFrame,
    log: Optional[MutableSequence[dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Parse Indonesian monthly inflation and standardize column names."""
    out = _rename_date_column(inflation)
    value_col = next(c for c in out.columns if "infl" in str(c).lower())
    out = out[["date", value_col]].rename(columns={value_col: "inflation_mom_pct"})
    out["inflation_mom_pct"] = _parse_indonesian_number(out["inflation_mom_pct"])
    out = out.sort_values("date").reset_index(drop=True)
    _assert_dates(out)
    assert not out["date"].duplicated().any()
    return out


def clean_trade_balance(
    trade_balance: pd.DataFrame,
    log: Optional[MutableSequence[dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Parse trade-balance values after removing Rp and thousands separators."""
    out = _rename_date_column(trade_balance)
    value_col = next(c for c in out.columns if c != "date")
    out = out[["date", value_col]].rename(columns={value_col: "neraca_ribu_usd"})
    out["neraca_ribu_usd"] = _parse_thousands_number(out["neraca_ribu_usd"])
    _record(log, "trade_balance", "unit_standardization",
            "Kolom disimpan sebagai neraca_ribu_usd setelah menghapus Rp dan koma pemisah ribuan.", len(out))
    out = out.sort_values("date").reset_index(drop=True)
    _assert_dates(out)
    assert not out["date"].duplicated().any()
    return out


def clean_pmi(
    pmi: pd.DataFrame,
    log: Optional[MutableSequence[dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Convert PMI quarter labels to quarter-end dates."""
    raw = pmi.copy()
    period_col = next(c for c in raw.columns if "quart" in str(c).lower() or "quarter" in str(c).lower())
    value_col = next(c for c in raw.columns if c != period_col)
    out = pd.DataFrame({
        "date": _quarter_end(raw[period_col]),
        "pmi_bi": pd.to_numeric(raw[value_col], errors="coerce"),
    }).sort_values("date").reset_index(drop=True)
    _assert_dates(out)
    assert not out["date"].duplicated().any()
    return out


def clean_bi_rate(
    bi_rate: pd.DataFrame,
    log: Optional[MutableSequence[dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Convert BI-7Day-RR decimals (e.g. 0.05) to percentage points (5.0)."""
    out = _rename_date_column(bi_rate)
    value_col = next(c for c in out.columns if c != "date")
    out = out[["date", value_col]].rename(columns={value_col: "bi_rate_pct"})
    out["bi_rate_pct"] = pd.to_numeric(out["bi_rate_pct"], errors="coerce") * 100
    out = out.sort_values("date").reset_index(drop=True)
    _assert_dates(out)
    assert not out["date"].duplicated().any()
    return out


def clean_summary_metrics(
    summary_metrics: pd.DataFrame,
    log: Optional[MutableSequence[dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Drop the aggregate ^JKLQ45 row and null ITMG metrics with incorrect scale."""
    out = summary_metrics.copy()
    before = len(out)
    out = out.loc[out["Ticker"].ne("^JKLQ45")].copy()
    _record(log, "summary_metrics", "drop_aggregate_row", "Menghapus Ticker ^JKLQ45.", before - len(out))
    bad_scale_cols = ["Price to Book (PBV)", "EV / EBITDA", "EV / Revenue"]
    mask = out["Ticker"].eq("ITMG.JK")
    affected = int(mask.sum() * len(bad_scale_cols))
    out.loc[mask, bad_scale_cols] = np.nan
    _record(log, "summary_metrics", "null_bad_scale_itmg",
            "ITMG.JK PBV, EV / EBITDA, dan EV / Revenue ditandai NaN karena salah skala.", affected)
    _assert_tickers(out)
    assert not out["Ticker"].duplicated().any()
    return out.reset_index(drop=True)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a processed DataFrame with ISO dates where applicable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)

