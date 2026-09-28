"""Simple static quality-value screening from ``summary_metrics.csv``.

This is a cross-sectional snapshot screen, not a historical trading signal.
The source metrics describe current/2026 conditions, so the resulting ranking
must only be used as an initial filter and must not be backfilled into earlier
periods.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_WEIGHTS = {
    "bank": {
        "roe_high": 0.40,
        "pbv_reasonable": 0.30,
        "pe_reasonable": 0.30,
    },
    "non_bank": {
        "roe_high": 0.35,
        "pbv_reasonable": 0.25,
        "pe_reasonable": 0.25,
        "de_low": 0.15,
    },
}


def _directional_score(values: pd.Series, higher_is_better: bool) -> pd.Series:
    """Convert peer-group ranks to 0-100, with 100 as best."""
    valid = values.dropna()
    if len(valid) <= 1:
        return pd.Series(100.0, index=valid.index)
    ranks = valid.rank(method="average", ascending=not higher_is_better)
    return 100 * (len(valid) - ranks) / (len(valid) - 1)


def _reasonable_score(values: pd.Series) -> pd.Series:
    """Score closeness to the peer-group median as valuation reasonableness."""
    valid = values.dropna()
    if len(valid) <= 1:
        return pd.Series(100.0, index=valid.index)
    median = valid.median()
    distance = (valid - median).abs()
    return _directional_score(distance, higher_is_better=False)


def _weighted_score(frame: pd.DataFrame, weights: dict[str, float]) -> tuple[pd.Series, pd.Series]:
    component_cols = list(weights)
    weighted_sum = pd.Series(0.0, index=frame.index)
    available_weight = pd.Series(0.0, index=frame.index)
    for component in component_cols:
        present = frame[component].notna()
        weighted_sum = weighted_sum.add(frame[component].fillna(0) * weights[component], fill_value=0)
        available_weight = available_weight.add(present.astype(float) * weights[component], fill_value=0)
    score = (weighted_sum / available_weight.replace(0, np.nan)).round(2)
    coverage = (100 * available_weight / sum(weights.values())).round(1)
    return score, coverage


def _add_group_scores(group: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    out = group.copy()
    out["score_roe_high"] = _directional_score(out["ROE (%)"], higher_is_better=True)
    out["score_pbv_reasonable"] = _reasonable_score(out["Price to Book (PBV)"])
    out["score_pe_reasonable"] = _reasonable_score(out["pe_used"])
    if "de_low" in weights:
        out["score_de_low"] = _directional_score(out["Debt to Equity"], higher_is_better=False)
    else:
        out["score_de_low"] = np.nan
    out["roe_high"] = out["score_roe_high"]
    out["pbv_reasonable"] = out["score_pbv_reasonable"]
    out["pe_reasonable"] = out["score_pe_reasonable"]
    out["de_low"] = out["score_de_low"]
    out["quality_score"], out["score_coverage_pct"] = _weighted_score(out, weights)
    return out


def build_quality_ranking(
    summary_metrics: pd.DataFrame,
    weights: dict[str, dict[str, float]] = DEFAULT_WEIGHTS,
) -> pd.DataFrame:
    """Build a 0-100 quality/value ranking with separate bank rules.

    Banks are identified by ``Industry`` containing ``Bank``. Bank scores use
    ROE, PBV, and P/E reasonableness only. Non-bank scores add an inverse
    Debt/Equity component. P/E uses trailing P/E, falling back to forward P/E
    when trailing P/E is missing. PBV and P/E reasonableness means closeness to
    the median of the relevant peer group, not simply the lowest multiple.
    """
    required = {
        "Ticker", "Industry", "ROE (%)", "Price to Book (PBV)",
        "Trailing P/E", "Forward P/E", "Debt to Equity",
    }
    missing = required.difference(summary_metrics.columns)
    if missing:
        raise KeyError(f"Kolom summary_metrics hilang: {sorted(missing)}")

    out = summary_metrics.copy()
    assert out["Ticker"].is_unique
    out["group"] = np.where(
        out["Industry"].astype(str).str.contains("Bank", case=False, na=False),
        "bank", "non_bank",
    )
    out["pe_used"] = out["Trailing P/E"].where(
        out["Trailing P/E"].notna(), out["Forward P/E"]
    )

    scored = []
    for group_name, group in out.groupby("group", sort=False):
        if group_name not in weights:
            raise KeyError(f"Weight belum tersedia untuk group {group_name!r}")
        scored.append(_add_group_scores(group, weights[group_name]))
    out = pd.concat(scored).sort_index()
    out["quality_score"] = out["quality_score"].round(2)
    out = out.sort_values(["quality_score", "score_coverage_pct", "Ticker"], ascending=[False, False, True])
    out["rank"] = np.arange(1, len(out) + 1)
    out["filter_use"] = "Filter awal statis snapshot 2026; bukan sinyal historis."

    result_cols = [
        "rank", "Ticker", "group", "quality_score", "score_coverage_pct",
        "ROE (%)", "Price to Book (PBV)", "pe_used", "Debt to Equity",
        "score_roe_high", "score_pbv_reasonable", "score_pe_reasonable", "score_de_low",
        "filter_use",
    ]
    result = out[result_cols].reset_index(drop=True)
    assert len(result) == 15
    assert result["rank"].tolist() == list(range(1, 16))
    assert result["Ticker"].is_unique
    assert result.loc[result["group"].eq("bank"), "score_de_low"].isna().all()
    return result


def build_quality_ranking_from_csv(
    input_path: Path | str = Path("data/processed/summary_metrics.csv"),
    output_path: Path | str = Path("data/processed/quality_score_ranking.csv"),
) -> pd.DataFrame:
    """Read the cleaned snapshot, rank 15 stocks, and write a CSV ranking."""
    ranking = build_quality_ranking(pd.read_csv(input_path))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output_path, index=False)
    return ranking


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    ranking = build_quality_ranking_from_csv(
        root / "data" / "processed" / "summary_metrics.csv",
        root / "data" / "processed" / "quality_score_ranking.csv",
    )
    print(ranking.to_string(index=False))
    print("\nCatatan: ranking ini adalah filter awal statis dari snapshot 2026, bukan sinyal historis.")

