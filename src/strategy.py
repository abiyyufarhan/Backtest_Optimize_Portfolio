"""Long-only, equal-weight trading strategies for the daily price panel.

The three strategy builders in this module return *executed weights*. A row at
date ``t`` is the portfolio held for the return from ``t`` to the next
observation. The underlying signal is calculated using the close at ``t-1``
and shifted one trading row forward, which models a signal observed at the
close of day ``t-1`` and execution at the price of day ``t``. This convention
avoids using the close that is being traded to form its own signal.

Only adjusted-close data is required. The module deliberately returns weights,
not P&L, because a realistic execution-price and transaction-cost model should
be selected by the caller. With close-only data, the next-row weights can be
combined with next-row close-to-close returns as a transparent approximation.

Strategies
----------
``ma_crossover_50_200``
    Long stocks when SMA(50) is above SMA(200).
``mean_reversion_bollinger_rsi``
    Long after a close at/below the lower Bollinger Band with RSI <= 30;
    exit when the close reaches the middle band or RSI >= 70.
``momentum_12_1_monthly``
    At each month-end, rank 12-month momentum excluding the latest month
    (252 trading days versus 21 trading days ago) and hold the top five until
    the next monthly rebalance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd


TRADING_DAYS_PER_MONTH = 21
TRADING_DAYS_PER_YEAR = 252
DEFAULT_TOP_N = 5


def _prepare_prices(prices: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return a sorted date-indexed price panel and its stock columns."""
    out = prices.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="raise")
        out = out.set_index("date")
    elif not isinstance(out.index, pd.DatetimeIndex):
        raise KeyError("prices harus memiliki kolom date atau DatetimeIndex.")

    out.index = pd.to_datetime(out.index, errors="raise")
    out = out.sort_index()
    assert not out.index.duplicated().any(), "Tanggal prices duplikat."
    assert out.index.is_monotonic_increasing, "Tanggal prices tidak monoton naik."

    stocks = [str(column) for column in out.columns if str(column).endswith(".JK")]
    if len(stocks) != 15:
        raise ValueError(f"Diharapkan 15 kolom saham .JK, ditemukan {len(stocks)}.")
    out[stocks] = out[stocks].apply(pd.to_numeric, errors="coerce")
    if (out[stocks] <= 0).any().any():
        raise ValueError("Harga saham harus positif atau NaN.")
    return out, stocks


def _equal_weight(selection: pd.DataFrame) -> pd.DataFrame:
    """Convert a boolean selection panel into row-wise equal weights."""
    selected = selection.astype(float)
    counts = selected.sum(axis=1)
    return selected.div(counts.replace(0, np.nan), axis=0).fillna(0.0)


def _execute_next_day(target_weights: pd.DataFrame) -> pd.DataFrame:
    """Shift close-based target weights to the next available trading row."""
    executed = target_weights.shift(1).fillna(0.0)
    assert executed.index.equals(target_weights.index)
    assert not executed.iloc[0].any(), "Hari pertama tidak boleh memiliki posisi." 
    assert np.isfinite(executed.to_numpy()).all()
    return executed


def _rsi(close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + relative_strength))


def ma_crossover_50_200(
    prices: pd.DataFrame,
    fast_window: int = 50,
    slow_window: int = 200,
) -> pd.DataFrame:
    """Return executed weights for the 50/200-day moving-average crossover.

    A stock is selected when its fast SMA is strictly above its slow SMA.
    Signals are calculated from the close at each date, then executed on the
    next available trading date. Rows before both moving averages exist are
    held in cash.
    """
    if fast_window >= slow_window:
        raise ValueError("fast_window harus lebih kecil dari slow_window.")
    panel, stocks = _prepare_prices(prices)
    close = panel[stocks]
    fast = close.rolling(fast_window, min_periods=fast_window).mean()
    slow = close.rolling(slow_window, min_periods=slow_window).mean()
    target = _equal_weight(fast.gt(slow) & fast.notna() & slow.notna())
    result = _execute_next_day(target)
    result.index.name = "date"
    return result


def mean_reversion_bollinger_rsi(
    prices: pd.DataFrame,
    bollinger_window: int = 20,
    bollinger_std: float = 2.0,
    rsi_window: int = 14,
    entry_rsi: float = 30.0,
    exit_rsi: float = 70.0,
) -> pd.DataFrame:
    """Return executed weights for a long-only Bollinger/RSI mean-reversion strategy.

    Entry requires both ``close <= lower_band`` and ``RSI <= entry_rsi``.
    Once held, a position remains active until ``close >= middle_band`` or
    ``RSI >= exit_rsi``. Each day's state is converted to equal weights and is
    shifted one trading row for next-day execution.
    """
    if bollinger_window < 2 or rsi_window < 2:
        raise ValueError("Window Bollinger dan RSI harus >= 2.")
    if bollinger_std <= 0 or not 0 < entry_rsi < exit_rsi < 100:
        raise ValueError("Parameter Bollinger/RSI tidak valid.")

    panel, stocks = _prepare_prices(prices)
    close = panel[stocks]
    middle = close.rolling(bollinger_window, min_periods=bollinger_window).mean()
    deviation = close.rolling(bollinger_window, min_periods=bollinger_window).std()
    lower = middle - bollinger_std * deviation
    rsi = _rsi(close, rsi_window)

    active = pd.DataFrame(False, index=close.index, columns=stocks)
    for row_number in range(len(close)):
        if row_number:
            active.iloc[row_number] = active.iloc[row_number - 1].to_numpy()
        entry = close.iloc[row_number].le(lower.iloc[row_number]) & rsi.iloc[row_number].le(entry_rsi)
        exit_ = close.iloc[row_number].ge(middle.iloc[row_number]) | rsi.iloc[row_number].ge(exit_rsi)
        active.iloc[row_number] = (active.iloc[row_number] & ~exit_) | entry
        active.iloc[row_number] &= middle.iloc[row_number].notna() & rsi.iloc[row_number].notna()

    target = _equal_weight(active)
    result = _execute_next_day(target)
    result.index.name = "date"
    return result


def momentum_12_1_monthly(
    prices: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
    lookback_days: int = TRADING_DAYS_PER_YEAR,
    skip_days: int = TRADING_DAYS_PER_MONTH,
) -> pd.DataFrame:
    """Return executed monthly-rebalanced weights for 12-1 momentum.

    At the last available trading row of each calendar month, momentum is
    ``close[t-skip_days] / close[t-lookback_days] - 1``. The latest month is
    therefore excluded. The top ``top_n`` valid stocks receive equal weights;
    the target is held until the next month-end signal and is executed on the
    next available trading row.
    """
    if not 0 < top_n <= 15:
        raise ValueError("top_n harus antara 1 dan 15.")
    if lookback_days <= skip_days:
        raise ValueError("lookback_days harus lebih besar dari skip_days.")

    panel, stocks = _prepare_prices(prices)
    close = panel[stocks]
    momentum = close.shift(skip_days).div(close.shift(lookback_days)).sub(1)
    month = pd.Series(close.index.to_period("M"), index=close.index)
    month_end = month.ne(month.shift(-1))

    monthly_targets = pd.DataFrame(np.nan, index=close.index, columns=stocks)
    for date in close.index[month_end.fillna(True)]:
        ranked = momentum.loc[date].dropna().nlargest(top_n)
        row = pd.Series(0.0, index=stocks)
        if not ranked.empty:
            row.loc[ranked.index] = 1.0 / len(ranked)
        monthly_targets.loc[date] = row

    target = monthly_targets.ffill().fillna(0.0)
    result = _execute_next_day(target)
    result.index.name = "date"
    return result


def build_strategy_weights(prices: pd.DataFrame) -> Mapping[str, pd.DataFrame]:
    """Build all three strategy weight panels using common input prices."""
    return {
        "ma_crossover_50_200": ma_crossover_50_200(prices),
        "mean_reversion_bollinger_rsi": mean_reversion_bollinger_rsi(prices),
        "momentum_12_1_monthly": momentum_12_1_monthly(prices),
    }


def assert_no_lookahead(
    strategy: Callable[[pd.DataFrame], pd.DataFrame],
    prices: pd.DataFrame,
    dates: int = 5,
) -> pd.DataFrame:
    """Check that executed weights through date ``t`` ignore observations after ``t``."""
    full, _ = _prepare_prices(prices)
    full_weights = strategy(full)
    eligible = full.index[1:]
    if len(eligible) < dates:
        raise AssertionError("Tidak cukup tanggal untuk no-lookahead test.")
    test_dates = eligible[np.linspace(0, len(eligible) - 1, dates, dtype=int)]
    rows = []
    for date in test_dates:
        truncated = full.loc[:date]
        truncated_weights = strategy(truncated)
        pd.testing.assert_frame_equal(
            full_weights.loc[:date], truncated_weights,
            check_exact=False, rtol=1e-12, atol=1e-12,
            check_dtype=False,
        )
        rows.append({"date": date, "unchanged_after_truncation": True})
    report = pd.DataFrame(rows)
    assert report["unchanged_after_truncation"].all()
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    prices = pd.read_csv(root / "data" / "processed" / "prices.csv", parse_dates=["date"])
    weights = build_strategy_weights(prices)
    for name, frame in weights.items():
        assert not frame.index.duplicated().any()
        print(f"{name}: shape={frame.shape}, nonzero_rows={(frame.sum(axis=1) > 0).sum()}")
        print(assert_no_lookahead(globals()[name], prices).to_string(index=False))
