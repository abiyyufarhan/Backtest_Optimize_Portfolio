"""Tests for point-in-time feature engineering."""

from pathlib import Path

import pandas as pd

from src.features import build_features, no_lookahead_test


def test_features_have_no_lookahead() -> None:
    project_root = Path(__file__).resolve().parents[1]
    master = pd.read_parquet(project_root / "data" / "processed" / "master_daily.parquet")
    report = no_lookahead_test(master, n_dates=5)
    assert len(report) == 5
    assert report["unchanged"].all()


def test_warmup_rows_are_removed() -> None:
    project_root = Path(__file__).resolve().parents[1]
    master = pd.read_parquet(project_root / "data" / "processed" / "master_daily.parquet")
    with_warmup = build_features(master, drop_warmup=False)
    without_warmup = build_features(master, drop_warmup=True)
    assert len(without_warmup) < len(with_warmup)
    assert without_warmup.isna().sum().sum() == 0

