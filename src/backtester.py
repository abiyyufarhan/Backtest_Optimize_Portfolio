"""Daily, long-only backtester with IDX-style transaction costs.

The public :func:`backtest` function accepts a signal matrix and a close-price
matrix. A signal observed on close ``t`` is shifted one trading row and traded
on the execution price of ``t+1``. Signals are interpreted as non-negative
scores: positive scores are normalized to fully invested equal/score-weighted
long positions, while an all-zero row stays in cash. Negative signals are
rejected because this backtester does not support shorting or leverage.

The default costs are deliberately configurable approximations for an IDX
retail account: 0.15% on buys, 0.25% on sells, and 0.05% slippage on either
side. Replace them with the user's broker schedule before relying on results.
The simulation reserves costs before placing a trade, so cash cannot become
negative through transaction costs.

``execution='close'`` uses the close price on ``t+1``. ``execution='open'``
requires ``open_prices`` and uses its price on ``t+1``; end-of-day equity is
then marked using the close-price matrix. No leverage is used in either mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252
DEFAULT_BUY_FEE = 0.0015
DEFAULT_SELL_FEE = 0.0025
DEFAULT_SLIPPAGE = 0.0005


@dataclass(frozen=True)
class BacktestConfig:
    """Execution and cost assumptions for :func:`backtest`."""

    initial_capital: float = 1_000_000.0
    buy_fee: float = DEFAULT_BUY_FEE
    sell_fee: float = DEFAULT_SELL_FEE
    slippage: float = DEFAULT_SLIPPAGE
    execution: str = "close"
    periods_per_year: int = TRADING_DAYS_PER_YEAR

    def validate(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital harus > 0.")
        if self.buy_fee < 0 or self.sell_fee < 0 or self.slippage < 0:
            raise ValueError("Fee dan slippage tidak boleh negatif.")
        if self.execution not in {"open", "close"}:
            raise ValueError("execution harus 'open' atau 'close'.")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year harus > 0.")


@dataclass
class BacktestResult:
    """Backtest outputs aligned on the daily trading calendar."""

    equity_curve: pd.DataFrame
    metrics: pd.DataFrame
    executed_weights: pd.DataFrame


def _as_date_index(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    out = frame.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="raise")
        out = out.set_index("date")
    elif not isinstance(out.index, pd.DatetimeIndex):
        raise KeyError(f"{name} harus memiliki kolom date atau DatetimeIndex.")
    out.index = pd.to_datetime(out.index, errors="raise")
    out = out.sort_index()
    assert not out.index.duplicated().any(), f"Tanggal {name} duplikat."
    assert out.index.is_monotonic_increasing, f"Tanggal {name} tidak monoton naik."
    return out


def _prepare_inputs(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    open_prices: Optional[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], list[str]]:
    signal_panel = _as_date_index(signals, "signals")
    close_panel = _as_date_index(prices, "prices")
    stock_columns = [str(column) for column in close_panel.columns if str(column).endswith(".JK")]
    if len(stock_columns) != 15:
        raise ValueError(f"Diharapkan 15 kolom saham .JK pada prices, ditemukan {len(stock_columns)}.")
    missing_signals = sorted(set(stock_columns).difference(signal_panel.columns))
    if missing_signals:
        raise KeyError(f"Signals tidak memiliki saham: {missing_signals}")

    dates = close_panel.index.intersection(signal_panel.index)
    if len(dates) < 2:
        raise ValueError("Minimal dua tanggal diperlukan untuk backtest.")
    close_panel = close_panel.loc[dates, stock_columns].apply(pd.to_numeric, errors="coerce")
    signal_panel = signal_panel.loc[dates, stock_columns].apply(pd.to_numeric, errors="coerce")
    if close_panel.isna().any().any() or (close_panel <= 0).any().any():
        raise ValueError("Prices mengandung NaN atau harga non-positif pada kalender backtest.")

    open_panel = None
    if open_prices is not None:
        open_panel = _as_date_index(open_prices, "open_prices")
        missing_open = sorted(set(stock_columns).difference(open_panel.columns))
        if missing_open:
            raise KeyError(f"open_prices tidak memiliki saham: {missing_open}")
        open_panel = open_panel.reindex(dates, columns=stock_columns).apply(pd.to_numeric, errors="coerce")
        if open_panel.isna().any().any() or (open_panel <= 0).any().any():
            raise ValueError("open_prices mengandung NaN atau harga non-positif.")

    signal_panel = signal_panel.fillna(0.0)
    if (signal_panel < 0).any().any():
        raise ValueError("Sinyal negatif tidak didukung: leverage/short selling dilarang.")
    return signal_panel, close_panel, open_panel, stock_columns


def _normalize_signals(signals: pd.DataFrame) -> pd.DataFrame:
    """Normalize positive signal scores to long-only target weights."""
    positive = signals.clip(lower=0.0)
    row_sum = positive.sum(axis=1)
    weights = positive.div(row_sum.replace(0.0, np.nan), axis=0).fillna(0.0)
    assert (weights >= 0).all().all()
    assert (weights.sum(axis=1) <= 1.0 + 1e-12).all()
    return weights


def _cash_after_trade(
    scale: float,
    cash_before: float,
    current_value: np.ndarray,
    desired_value: np.ndarray,
    buy_rate: float,
    sell_rate: float,
) -> tuple[float, np.ndarray, np.ndarray, float, float]:
    target_value = scale * desired_value
    delta = target_value - current_value
    buy_value = np.clip(delta, 0.0, None)
    sell_value = np.clip(-delta, 0.0, None)
    buy_cost = float(buy_value.sum() * buy_rate)
    sell_cost = float(sell_value.sum() * sell_rate)
    cash_after = float(cash_before - buy_value.sum() + sell_value.sum() - buy_cost - sell_cost)
    return cash_after, target_value, buy_value, sell_value, buy_cost + sell_cost


def _feasible_trade(
    cash_before: float,
    current_value: np.ndarray,
    desired_value: np.ndarray,
    buy_rate: float,
    sell_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Scale a desired trade down, if needed, to keep post-trade cash >= 0."""
    at_one = _cash_after_trade(1.0, cash_before, current_value, desired_value, buy_rate, sell_rate)
    if at_one[0] >= -1e-9:
        scale = 1.0
    else:
        low, high = 0.0, 1.0
        for _ in range(70):
            mid = (low + high) / 2.0
            if _cash_after_trade(mid, cash_before, current_value, desired_value, buy_rate, sell_rate)[0] >= 0:
                low = mid
            else:
                high = mid
        scale = low
    cash_after, target_value, buy_value, sell_value, total_cost = _cash_after_trade(
        scale, cash_before, current_value, desired_value, buy_rate, sell_rate,
    )
    assert cash_after >= -1e-7, "Trade menyebabkan cash negatif."
    return target_value, buy_value, sell_value, cash_after, total_cost, scale


def _risk_free_daily(master_daily: pd.DataFrame, dates: pd.DatetimeIndex, periods_per_year: int) -> pd.Series:
    master = _as_date_index(master_daily, "master_daily")
    if "bi_rate" not in master.columns:
        raise KeyError("master_daily harus memiliki kolom bi_rate.")
    annual_rate_pct = pd.to_numeric(master["bi_rate"], errors="coerce").reindex(dates).ffill()
    if annual_rate_pct.isna().any():
        raise ValueError("bi_rate tidak tersedia pada awal kalender backtest.")
    return (1.0 + annual_rate_pct / 100.0).pow(1.0 / periods_per_year) - 1.0


def _benchmark_equity(
    close_prices: pd.DataFrame,
    master_daily: pd.DataFrame,
    stock_columns: list[str],
    initial_capital: float,
) -> tuple[pd.Series, pd.Series]:
    master = _as_date_index(master_daily, "master_daily")
    if "lq45" not in master.columns:
        raise KeyError("master_daily harus memiliki kolom lq45 untuk benchmark.")
    lq45 = pd.to_numeric(master["lq45"], errors="coerce").reindex(close_prices.index).ffill()
    if lq45.isna().any() or (lq45 <= 0).any():
        raise ValueError("Benchmark lq45 tidak lengkap pada kalender backtest.")

    first_prices = close_prices.iloc[0]
    equal_weight_equity = initial_capital * (close_prices.div(first_prices, axis=1).mean(axis=1))
    lq45_equity = initial_capital * lq45 / lq45.iloc[0]
    return lq45_equity.rename("benchmark_lq45"), equal_weight_equity.rename("benchmark_equal_weight")


def cagr(equity: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Compound annual growth rate from an equity curve."""
    equity = pd.Series(equity).dropna()
    if len(equity) < 2 or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return np.nan
    years = (len(equity) - 1) / periods_per_year
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualized standard deviation of daily returns."""
    returns = pd.Series(returns).dropna()
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year)) if len(returns) > 1 else np.nan


def sharpe_ratio(
    returns: pd.Series,
    risk_free_daily: Optional[pd.Series] = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio using daily BI Rate-derived excess returns."""
    returns = pd.Series(returns).astype(float)
    rf = pd.Series(0.0, index=returns.index) if risk_free_daily is None else pd.Series(risk_free_daily).reindex(returns.index)
    excess = (returns - rf).dropna()
    denominator = excess.std(ddof=1)
    return float(excess.mean() / denominator * np.sqrt(periods_per_year)) if len(excess) > 1 and denominator > 0 else np.nan


def sortino_ratio(
    returns: pd.Series,
    risk_free_daily: Optional[pd.Series] = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sortino ratio using downside deviation of excess returns."""
    returns = pd.Series(returns).astype(float)
    rf = pd.Series(0.0, index=returns.index) if risk_free_daily is None else pd.Series(risk_free_daily).reindex(returns.index)
    excess = (returns - rf).dropna()
    downside = np.minimum(excess.to_numpy(), 0.0)
    downside_deviation = np.sqrt(np.mean(downside ** 2)) if len(downside) else np.nan
    return float(excess.mean() / downside_deviation * np.sqrt(periods_per_year)) if downside_deviation > 0 else np.nan


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a negative decimal."""
    equity = pd.Series(equity).dropna()
    if equity.empty:
        return np.nan
    return float((equity / equity.cummax() - 1.0).min())


def calmar_ratio(equity: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """CAGR divided by absolute maximum drawdown."""
    drawdown = max_drawdown(equity)
    return float(cagr(equity, periods_per_year) / abs(drawdown)) if drawdown < 0 else np.nan


def win_rate(returns: pd.Series) -> float:
    """Fraction of non-zero return observations that are profitable."""
    active = pd.Series(returns).dropna()
    active = active[active != 0]
    return float((active > 0).mean()) if len(active) else np.nan


def turnover(trade_turnover: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualized gross turnover; one full sell plus buy counts as 2x."""
    values = pd.Series(trade_turnover).dropna()
    return float(values.mean() * periods_per_year) if len(values) else np.nan


def calculate_metrics(
    equity: pd.Series,
    risk_free_daily: Optional[pd.Series] = None,
    trade_turnover: Optional[pd.Series] = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """Calculate the requested risk and trading metrics for one equity curve."""
    equity = pd.Series(equity).sort_index()
    returns = equity.pct_change().fillna(0.0)
    result = {
        "CAGR": cagr(equity, periods_per_year),
        "volatility_annual": annualized_volatility(returns, periods_per_year),
        "Sharpe": sharpe_ratio(returns, risk_free_daily, periods_per_year),
        "Sortino": sortino_ratio(returns, risk_free_daily, periods_per_year),
        "Max Drawdown": max_drawdown(equity),
        "Calmar": calmar_ratio(equity, periods_per_year),
        "win_rate": win_rate(returns),
        "turnover_annual": turnover(trade_turnover, periods_per_year) if trade_turnover is not None else 0.0,
    }
    return result


def backtest(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    master_daily: pd.DataFrame,
    *,
    open_prices: Optional[pd.DataFrame] = None,
    config: BacktestConfig = BacktestConfig(),
) -> BacktestResult:
    """Run a daily long-only backtest and return equity curves and metrics.

    Parameters
    ----------
    signals:
        Date-indexed or date-column matrix with the 15 ``.JK`` columns. A
        positive row is normalized into target long weights; all zeros means
        cash. The row at ``t`` is executed at ``t+1``.
    prices:
        Close-price matrix containing the same 15 stock columns.
    master_daily:
        Point-in-time master containing ``date``, ``lq45``, and ``bi_rate``.
        ``bi_rate`` is interpreted as an annual percentage and converted to a
        daily risk-free rate for Sharpe and Sortino.
    open_prices:
        Optional open-price matrix. Required only when ``config.execution`` is
        ``'open'``. The equity curve is marked at close prices.
    config:
        Fees, slippage, execution mode, starting capital, and annualization.
    """
    config.validate()
    signal_panel, close_panel, open_panel, stock_columns = _prepare_inputs(signals, prices, open_prices)
    if config.execution == "open" and open_panel is None:
        raise ValueError("open_prices wajib diberikan untuk execution='open'.")

    target_weights = _normalize_signals(signal_panel)
    executed_weights = target_weights.shift(1).fillna(0.0)
    execution_prices = close_panel if config.execution == "close" else open_panel
    assert execution_prices is not None

    cash = float(config.initial_capital)
    shares = np.zeros(len(stock_columns), dtype=float)
    rows: list[dict[str, float | pd.Timestamp]] = []
    buy_rate = config.buy_fee + config.slippage
    sell_rate = config.sell_fee + config.slippage

    for row_number, date in enumerate(close_panel.index):
        trade_price = execution_prices.loc[date].to_numpy(dtype=float)
        mark_price = close_panel.loc[date].to_numpy(dtype=float)
        current_value = shares * trade_price
        pre_trade_value = float(cash + current_value.sum())
        desired_value = executed_weights.loc[date].to_numpy(dtype=float) * pre_trade_value
        target_value, buy_value, sell_value, cash, total_cost, scale = _feasible_trade(
            cash, current_value, desired_value, buy_rate, sell_rate,
        )
        shares = target_value / trade_price
        end_value = float(cash + (shares * mark_price).sum())
        gross_exposure = float((shares * mark_price).sum() / end_value) if end_value else 0.0
        gross_turnover = float((buy_value.sum() + sell_value.sum()) / pre_trade_value) if pre_trade_value else 0.0
        rows.append({
            "date": date,
            "equity": end_value,
            "cash": cash,
            "gross_exposure": gross_exposure,
            "buy_value": float(buy_value.sum()),
            "sell_value": float(sell_value.sum()),
            "transaction_cost": total_cost,
            "turnover_daily": gross_turnover,
            "trade_scale": scale,
        })

    curve = pd.DataFrame(rows).set_index("date")
    curve["daily_return"] = curve["equity"].pct_change().fillna(0.0)
    curve["risk_free_daily"] = _risk_free_daily(master_daily, curve.index, config.periods_per_year)
    lq45_equity, equal_weight_equity = _benchmark_equity(
        close_panel, master_daily, stock_columns, config.initial_capital,
    )
    curve["benchmark_lq45"] = lq45_equity
    curve["benchmark_equal_weight"] = equal_weight_equity

    metric_rows = []
    for name, equity in {
        "strategy": curve["equity"],
        "benchmark_lq45": curve["benchmark_lq45"],
        "benchmark_equal_weight": curve["benchmark_equal_weight"],
    }.items():
        metrics = calculate_metrics(
            equity,
            risk_free_daily=curve["risk_free_daily"],
            trade_turnover=curve["turnover_daily"] if name == "strategy" else None,
            periods_per_year=config.periods_per_year,
        )
        metric_rows.append({"portfolio": name, **metrics})
    metrics_frame = pd.DataFrame(metric_rows).set_index("portfolio")
    assert curve.index.is_monotonic_increasing and not curve.index.duplicated().any()
    assert (curve["cash"] >= -1e-6).all(), "Cash negatif: leverage terdeteksi."
    assert (executed_weights.sum(axis=1) <= 1.0 + 1e-12).all()
    return BacktestResult(curve, metrics_frame, executed_weights)


def backtest_from_csv(
    signals_path: Path | str,
    prices_path: Path | str,
    master_path: Path | str,
    output_path: Optional[Path | str] = None,
    config: BacktestConfig = BacktestConfig(),
) -> BacktestResult:
    """Convenience wrapper for CSV inputs; Parquet can be loaded by the caller."""
    signals = pd.read_csv(signals_path, parse_dates=["date"])
    prices = pd.read_csv(prices_path, parse_dates=["date"])
    master = pd.read_parquet(master_path)
    result = backtest(signals, prices, master, config=config)
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.equity_curve.to_csv(output)
    return result

