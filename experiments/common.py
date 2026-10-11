"""Shared rolling-origin setup for every forecaster."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SERIES_PATH = ROOT / "data" / "selected_series.npz"
FUNCTIONS_PATH = ROOT / "data" / "functions.csv"
OUT_DIR = ROOT / "results"

MINUTES_PER_DAY = 1440
# One full day. Seasonal naive already reads this lag. Last-value and the
# trailing percentile only use the last 1 and 10 minutes, so this length does
# not change them. ETS, the LSTM, and Chronos all receive this window.
CONTEXT = 1440
HORIZON = 60
STRIDE = 60
TEST_START = 10 * MINUTES_PER_DAY
TEST_END = 14 * MINUTES_PER_DAY
HELD_OUT_START = 12 * MINUTES_PER_DAY
DAILY_ACF_MIN = 0.30


def load_series() -> tuple[np.ndarray, pd.DataFrame]:
    bundle = np.load(SERIES_PATH)
    series = bundle["series"]
    ids = bundle["function_id"].astype(str)
    meta = pd.read_csv(FUNCTIONS_PATH)
    meta["function_id"] = meta["function_id"].astype(str)
    meta = meta.set_index("function_id").loc[ids].reset_index()
    if len(meta) != len(series):
        raise SystemExit("Function list does not match the series file")
    return series, meta


def origins() -> np.ndarray:
    starts = np.arange(TEST_START, TEST_END, STRIDE)
    if starts[0] < CONTEXT or starts[-1] + HORIZON != TEST_END:
        raise SystemExit("Origin grid does not cover days 11-14")
    return starts


def gather(series: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    index = starts[:, None] + np.arange(length)[None, :]
    return series[:, index]


def origin_slices(starts: np.ndarray) -> dict[str, slice]:
    held_out = int(np.searchsorted(starts, HELD_OUT_START))
    return {
        "days_11_14": slice(None),
        "days_11_12": slice(0, held_out),
        "days_13_14": slice(held_out, None),
    }


def check_coverage(y: np.ndarray, meta: pd.DataFrame) -> None:
    covered = np.rint(y.reshape(len(y), -1).sum(axis=1))
    expected = np.rint(meta["test_invocations"].to_numpy())
    if not np.array_equal(covered, expected):
        raise SystemExit("Forecast windows do not cover the day 11-14 invocation totals")
