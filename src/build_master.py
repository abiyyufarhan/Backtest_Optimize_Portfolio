"""Build a point-in-time daily master dataset for backtesting.

The release-date model is an approximation: it applies configurable business-day
lags to period-end dates. It does not know the actual publication calendar,
embargo time, revisions, or late releases. Change ``RELEASE_LAGS`` when verified
release-calendar information becomes available. ``pd.merge_asof`` with
``direction='backward'`` ensures a price date only sees observations whose
modeled ``tanggal_rilis`` is on or before that price date.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay


# One central, editable configuration. All values are business-day lags from
# the processed observation date. BI Rate dates are already release dates.
RELEASE_LAGS: dict[str, int] = {
    "bi_rate": 0,
    "inflasi_mtm": 5,
    "neraca": 15,
    "pdb_yoy": 35,
    "pmi_bi": 5,
}

MACRO_SOURCES: dict[str, dict[str, str]] = {
    "bi_rate": {"file": "bi_rate.csv", "value_col": "bi_rate_pct"},
    "inflasi_mtm": {"file": "inflation.csv", "value_col": "inflation_mom_pct"},
    "neraca": {"file": "trade_balance.csv", "value_col": "neraca_ribu_usd"},
    "pdb_yoy": {"file": "pdb.csv", "value_col": "pertumbuhan_yoy_pct"},
    "pmi_bi": {"file": "pmi_bi.csv", "value_col": "pmi_bi"},
}

MASTER_MACRO_COLUMNS = list(MACRO_SOURCES)


def _read_csv(processed_dir: Path, filename: str) -> pd.DataFrame:
    path = processed_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset tidak ditemukan: {path}")
    out = pd.read_csv(path)
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="raise")
    return out


def _assert_dates(df: pd.DataFrame, date_col: str = "date") -> None:
    assert date_col in df.columns
    assert not df[date_col].duplicated().any(), f"Tanggal duplikat pada {date_col}."
    assert df[date_col].is_monotonic_increasing, f"Tanggal {date_col} tidak monoton naik."


def add_release_date(
    frame: pd.DataFrame,
    dataset: str,
    release_lags: Mapping[str, int] = RELEASE_LAGS,
) -> pd.DataFrame:
    """Add modeled ``tanggal_rilis`` using a configurable business-day lag."""
    if dataset not in release_lags:
        raise KeyError(f"Lag belum dikonfigurasi untuk dataset {dataset!r}.")
    if "date" not in frame.columns:
        raise KeyError("Dataset makro harus memiliki kolom date.")
    if not isinstance(release_lags[dataset], int) or release_lags[dataset] < 0:
        raise ValueError(f"Lag {dataset} harus integer business days >= 0.")

    out = frame.copy().sort_values("date").reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    out["tanggal_rilis"] = out["date"] + BDay(release_lags[dataset])
    _assert_dates(out)
    assert out["tanggal_rilis"].is_monotonic_increasing
    assert (out["tanggal_rilis"] >= out["date"]).all()
    return out


def _merge_macro_asof(
    master: pd.DataFrame,
    macro: pd.DataFrame,
    dataset: str,
    value_col: str,
    release_lags: Mapping[str, int],
) -> pd.DataFrame:
    prepared = add_release_date(macro[["date", value_col]], dataset, release_lags)
    prepared = prepared.rename(columns={value_col: dataset})
    release_col = f"{dataset}_tanggal_rilis"
    prepared = prepared.rename(columns={"tanggal_rilis": release_col})
    prepared = prepared[[release_col, dataset]].sort_values(release_col)
    assert not prepared[release_col].duplicated().any(), f"Release date duplikat pada {dataset}."

    out = pd.merge_asof(
        master.sort_values("date"),
        prepared,
        left_on="date",
        right_on=release_col,
        direction="backward",
    )
    _assert_dates(out)
    assert (out[release_col].dropna() <= out.loc[out[release_col].notna(), "date"]).all()
    return out


def build_master(
    processed_dir: Path | str = Path("data/processed"),
    output_path: Path | str = Path("data/processed/master_daily.parquet"),
    release_lags: Mapping[str, int] = RELEASE_LAGS,
    include_release_dates: bool = False,
) -> pd.DataFrame:
    """Build and optionally write the point-in-time daily master dataset.

    Parameters
    ----------
    processed_dir:
        Directory containing the CSV outputs from ``02_cleaning.ipynb``.
    output_path:
        Parquet destination. A Parquet engine such as ``pyarrow`` is required.
    release_lags:
        Business-day lag from each processed observation date to modeled release.
    include_release_dates:
        Keep audit columns named ``<macro>_tanggal_rilis`` in the returned frame.
        The persisted default output omits these audit columns.

    Notes
    -----
    The modeled lag is not a historical release calendar. It does not account
    for actual publication times, weekends/holidays beyond pandas' business-day
    offset, revisions, embargoes, or data-vintage changes. Use the configuration
    as an explicit assumption for backtesting and replace it with verified
    release timestamps when available.
    """
    processed_dir = Path(processed_dir)
    output_path = Path(output_path)
    prices = _read_csv(processed_dir, "prices.csv")
    index_lq45 = _read_csv(processed_dir, "index_lq45.csv")
    usdidr = _read_csv(processed_dir, "usdidr.csv")
    _assert_dates(prices)
    _assert_dates(index_lq45)
    _assert_dates(usdidr)

    price_cols = [c for c in prices.columns if c != "date"]
    assert len(price_cols) == 15, f"Diharapkan 15 kolom saham, ditemukan {len(price_cols)}."
    master = prices.copy()
    master = master.merge(
        index_lq45[["date", "adj_close"]].rename(columns={"adj_close": "lq45"}),
        on="date", how="left", validate="one_to_one",
    )
    master = master.merge(
        usdidr[["date", "adj_close"]].rename(columns={"adj_close": "usdidr"}),
        on="date", how="left", validate="one_to_one",
    ).sort_values("date").reset_index(drop=True)
    _assert_dates(master)

    for dataset, spec in MACRO_SOURCES.items():
        macro = _read_csv(processed_dir, spec["file"])
        _assert_dates(macro)
        master = _merge_macro_asof(master, macro, dataset, spec["value_col"], release_lags)

    expected = ["date", *price_cols, "lq45", "usdidr", *MASTER_MACRO_COLUMNS]
    release_cols = [f"{name}_tanggal_rilis" for name in MASTER_MACRO_COLUMNS]
    audit_columns = [c for c in release_cols if c in master.columns]
    assert set(expected).issubset(master.columns)
    assert master[expected].shape[1] == len(expected)
    _assert_dates(master)

    result = master[expected + (audit_columns if include_release_dates else [])].copy()
    if not include_release_dates:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            result.to_parquet(output_path, index=False)
        except (ImportError, ModuleNotFoundError) as exc:
            fallback = output_path.with_suffix(".csv")
            result.to_csv(fallback, index=False)
            raise RuntimeError(
                f"Parquet engine tidak tersedia. Fallback CSV ditulis ke {fallback}; "
                "install pyarrow atau fastparquet untuk membuat master_daily.parquet."
            ) from exc
    return result


def run_release_lag_tests(
    master_with_release_dates: pd.DataFrame,
    n: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Print five deterministic point-in-time checks and assert release ordering."""
    release_cols = [f"{name}_tanggal_rilis" for name in MASTER_MACRO_COLUMNS]
    value_cols = MASTER_MACRO_COLUMNS
    eligible = master_with_release_dates.dropna(subset=value_cols + release_cols).copy()
    if len(eligible) < n:
        raise AssertionError("Tidak cukup tanggal dengan seluruh data makro untuk sampling test.")
    sample = eligible.sample(n=n, random_state=random_state).sort_values("date")
    rows: list[dict[str, Any]] = []
    for _, row in sample.iterrows():
        for value_col, release_col in zip(value_cols, release_cols):
            release_date = pd.Timestamp(row[release_col])
            price_date = pd.Timestamp(row["date"])
            assert release_date <= price_date, f"Look-ahead terdeteksi pada {value_col}."
            rows.append({
                "tanggal_harga": price_date,
                "dataset": value_col,
                "nilai_terpakai": row[value_col],
                "tanggal_rilis": release_date,
                "release_le_price": release_date <= price_date,
            })
    report = pd.DataFrame(rows)
    print("Release-lag test sample:")
    print(report.to_string(index=False))
    assert report["release_le_price"].all()
    return report


def main() -> None:
    processed_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    audit_master = build_master(
        processed_dir=processed_dir,
        output_path=processed_dir / "master_daily.parquet",
        include_release_dates=True,
    )
    run_release_lag_tests(audit_master)
    build_master(
        processed_dir=processed_dir,
        output_path=processed_dir / "master_daily.parquet",
        include_release_dates=False,
    )
    print(f"Master dataset dibuat: {processed_dir / 'master_daily.parquet'}")


if __name__ == "__main__":
    main()

