"""Point-in-time feature engineering for ``master_daily.parquet``.

Every feature is calculated with observations available on or before day ``t``.
There is intentionally no negative shift, centered rolling window, or future
merge in this module. The 1/3/6/12-month momentum windows use 21/63/126/252
trading days. Rolling volatility is 20-day annualized volatility.

The master file contains adjusted close for individual stocks but no individual
stock high/low columns. Therefore ``<ticker>_atr14`` uses a close-to-close
true-range proxy: ``abs(close[t] - close[t-1])``. Pass an OHLC mapping to
``build_features`` when true high-low ATR data is available.

``inflasi_3m`` is a trailing 63-trading-day average of the as-of monthly
inflation signal. The daily master repeats the latest released macro value, so
this definition preserves point-in-time behavior without treating one monthly
observation as 63 independent monthly releases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd


MOMENTUM_WINDOWS = {
    "1m": 21,
    "3m": 63,
    "6m": 126,
    "12m": 252,
}
SMA_WINDOWS = (20, 50, 200)
EMA_WINDOWS = (20, 50, 200)
RSI_WINDOW = 14
ATR_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_WINDOW = 20
VOLATILITY_WINDOW = 20
TRADING_DAYS_PER_YEAR = 252


def _prepare_master(master: pd.DataFrame) -> pd.DataFrame:
    out = master.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="raise")
    elif isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index().rename(columns={out.index.name or "index": "date"})
    else:
        raise KeyError("Master dataset harus memiliki kolom date atau DatetimeIndex.")
    out = out.sort_values("date").reset_index(drop=True)
    assert not out["date"].duplicated().any(), "Tanggal master duplikat."
    assert out["date"].is_monotonic_increasing, "Tanggal master tidak monoton naik."
    return out


def _stock_columns(master: pd.DataFrame) -> list[str]:
    stocks = [c for c in master.columns if str(c).endswith(".JK")]
    if len(stocks) != 15:
        raise ValueError(f"Diharapkan 15 kolom saham .JK, ditemukan {len(stocks)}.")
    return stocks


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(
    close: pd.Series,
    window: int,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
) -> pd.Series:
    previous_close = close.shift(1)
    if high is not None and low is not None:
        true_range = pd.concat([
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ], axis=1).max(axis=1)
    else:
        true_range = close.diff().abs()
    return true_range.rolling(window, min_periods=window).mean()


def _feature_names(stocks: Sequence[str]) -> list[str]:
    names: list[str] = []
    for ticker in stocks:
        names.extend([f"{ticker}_sma{w}" for w in SMA_WINDOWS])
        names.extend([f"{ticker}_ema{w}" for w in EMA_WINDOWS])
        names.append(f"{ticker}_rsi{RSI_WINDOW}")
        names.extend([
            f"{ticker}_macd",
            f"{ticker}_macd_signal",
            f"{ticker}_macd_hist",
            f"{ticker}_bb_mid",
            f"{ticker}_bb_upper",
            f"{ticker}_bb_lower",
            f"{ticker}_atr{ATR_WINDOW}",
            f"{ticker}_volatility{VOLATILITY_WINDOW}",
        ])
        names.extend([f"{ticker}_momentum_{label}" for label in MOMENTUM_WINDOWS])
    names.extend([
        "bi_rate_change",
        "usdidr_change_20d",
        "pmi_bi_expansion",
        "inflasi_3m",
    ])
    return names


def build_features(
    master: pd.DataFrame,
    ohlc: Optional[Mapping[str, Mapping[str, pd.Series]]] = None,
    drop_warmup: bool = True,
) -> pd.DataFrame:
    """Calculate point-in-time technical and macro features.

    Parameters
    ----------
    master:
        Daily master with ``date``, 15 ``.JK`` price columns, ``lq45``,
        ``usdidr``, ``bi_rate``, ``inflasi_mtm``, ``pmi_bi`` and other macro
        columns.
    ohlc:
        Optional mapping such as ``{'BBCA.JK': {'high': ..., 'low': ...}}``.
        If absent, ATR uses the close-to-close proxy documented above.
    drop_warmup:
        Drop rows containing NaN in any generated feature. Set False when
        running the no-lookahead test so early warm-up rows remain visible.
    """
    base = _prepare_master(master)
    stocks = _stock_columns(base)
    work = base.set_index("date")
    ohlc = ohlc or {}
    feature_values: dict[str, pd.Series] = {}

    for ticker in stocks:
        close = work[ticker]
        for window in SMA_WINDOWS:
            feature_values[f"{ticker}_sma{window}"] = close.rolling(window, min_periods=window).mean()
        for window in EMA_WINDOWS:
            feature_values[f"{ticker}_ema{window}"] = close.ewm(span=window, adjust=False, min_periods=window).mean()

        feature_values[f"{ticker}_rsi{RSI_WINDOW}"] = _rsi(close, RSI_WINDOW)
        ema_fast = close.ewm(span=MACD_FAST, adjust=False, min_periods=MACD_FAST).mean()
        ema_slow = close.ewm(span=MACD_SLOW, adjust=False, min_periods=MACD_SLOW).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=MACD_SIGNAL, adjust=False, min_periods=MACD_SIGNAL).mean()
        feature_values[f"{ticker}_macd"] = macd
        feature_values[f"{ticker}_macd_signal"] = signal
        feature_values[f"{ticker}_macd_hist"] = macd - signal

        bb_mid = close.rolling(BOLLINGER_WINDOW, min_periods=BOLLINGER_WINDOW).mean()
        bb_std = close.rolling(BOLLINGER_WINDOW, min_periods=BOLLINGER_WINDOW).std()
        feature_values[f"{ticker}_bb_mid"] = bb_mid
        feature_values[f"{ticker}_bb_upper"] = bb_mid + 2 * bb_std
        feature_values[f"{ticker}_bb_lower"] = bb_mid - 2 * bb_std

        high = low = None
        if ticker in ohlc and {"high", "low"}.issubset(ohlc[ticker]):
            high = pd.Series(ohlc[ticker].get("high"), index=work.index)
            low = pd.Series(ohlc[ticker].get("low"), index=work.index)
        feature_values[f"{ticker}_atr{ATR_WINDOW}"] = _atr(close, ATR_WINDOW, high, low)
        feature_values[f"{ticker}_volatility{VOLATILITY_WINDOW}"] = (
            np.log(close / close.shift(1))
            .rolling(VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW)
            .std()
            * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
        for label, window in MOMENTUM_WINDOWS.items():
            feature_values[f"{ticker}_momentum_{label}"] = close.pct_change(window)

    feature_values["bi_rate_change"] = work["bi_rate"].diff()
    feature_values["usdidr_change_20d"] = work["usdidr"].pct_change(20)
    feature_values["pmi_bi_expansion"] = (work["pmi_bi"] > 50).astype("int8")
    feature_values["inflasi_3m"] = (
        work["inflasi_mtm"]
        .rolling(63, min_periods=63)
        .mean()
    )

    work = pd.concat([work, pd.DataFrame(feature_values, index=work.index)], axis=1)
    out = work.reset_index()
    feature_cols = _feature_names(stocks)
    assert set(feature_cols).issubset(out.columns)
    assert not out["date"].duplicated().any()
    assert out["date"].is_monotonic_increasing
    if drop_warmup:
        out = out.dropna(subset=feature_cols).reset_index(drop=True)
        assert not out[feature_cols].isna().any().any()
    return out


def no_lookahead_test(
    master: pd.DataFrame,
    n_dates: int = 5,
    dates: Optional[Sequence[pd.Timestamp]] = None,
) -> pd.DataFrame:
    """Prove features at t are unchanged when observations after t are removed."""
    base = _prepare_master(master)
    full = build_features(base, drop_warmup=False).set_index("date")
    feature_cols = _feature_names(_stock_columns(base))
    eligible = full.dropna(subset=feature_cols).index
    if dates is None:
        if len(eligible) < n_dates:
            raise AssertionError("Tidak cukup tanggal setelah warm-up untuk test.")
        positions = np.linspace(0, len(eligible) - 1, n_dates, dtype=int)
        dates = [eligible[position] for position in positions]
    dates = [pd.Timestamp(date) for date in dates]

    report_rows = []
    for date in dates:
        if date not in full.index:
            raise AssertionError(f"Tanggal test tidak ada: {date.date()}")
        truncated = base.loc[base["date"] <= date]
        truncated_features = build_features(truncated, drop_warmup=False).set_index("date")
        left = full.loc[date, feature_cols]
        right = truncated_features.loc[date, feature_cols]
        pd.testing.assert_series_equal(
            left, right, check_exact=False, rtol=1e-12, atol=1e-12,
            check_names=False, check_dtype=False,
        )
        report_rows.append({
            "date": date,
            "features_checked": len(feature_cols),
            "future_rows_removed": len(base) - len(truncated),
            "unchanged": True,
        })
    report = pd.DataFrame(report_rows)
    assert report["unchanged"].all()
    return report


def build_features_from_parquet(
    input_path: Path | str = Path("data/processed/master_daily.parquet"),
    output_path: Path | str = Path("data/processed/master_features.parquet"),
) -> pd.DataFrame:
    """Read master Parquet, build features, write processed feature Parquet."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    master = pd.read_parquet(input_path)
    features = build_features(master, drop_warmup=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    return features


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    master_path = root / "data" / "processed" / "master_daily.parquet"
    feature_path = root / "data" / "processed" / "master_features.parquet"
    master = pd.read_parquet(master_path)
    feature_frame = build_features(master, drop_warmup=True)
    report = no_lookahead_test(master)
    feature_frame.to_parquet(feature_path, index=False)
    print(f"Features written: {feature_path} | shape={feature_frame.shape}")
    print(report.to_string(index=False))

