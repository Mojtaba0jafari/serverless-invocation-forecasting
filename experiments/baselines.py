"""Rolling-origin forecasts for the three cheap baselines.

Context is the previous 360 minutes. The horizon is the next 60 minutes.
A new forecast starts every 60 minutes across days 11-14, so each test minute
is scored once. Days 1-10 are history only.

Seasonal naive repeats the same hour yesterday when the training day-lag
autocorrelation is above 0.30. Otherwise it repeats the previous 60 minutes.
Last-value repeats the latest observed minute. Trailing percentile holds the
linear 95th percentile of the previous 10 minutes constant.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import WINDOWS, class_table, per_function_table, self_check

ROOT = Path(__file__).resolve().parents[1]
SERIES_PATH = ROOT / "data" / "selected_series.npz"
FUNCTIONS_PATH = ROOT / "data" / "functions.csv"
OUT_DIR = ROOT / "results"

MINUTES_PER_DAY = 1440
CONTEXT = 360
HORIZON = 60
STRIDE = 60
TEST_START = 10 * MINUTES_PER_DAY
TEST_END = 14 * MINUTES_PER_DAY
HELD_OUT_START = 12 * MINUTES_PER_DAY
DAILY_ACF_MIN = 0.30
TRAILING_MINUTES = 10
TRAILING_QUANTILE = 0.95
MODELS = ("seasonal_naive", "last_value", "trailing_percentile")


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
    if np.any(starts < MINUTES_PER_DAY):
        raise SystemExit("Seasonal lag would read before the start of the trace")
    return starts


def gather(series: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    index = starts[:, None] + np.arange(length)[None, :]
    return series[:, index]


def forecast(series: np.ndarray, meta: pd.DataFrame, starts: np.ndarray) -> dict[str, np.ndarray]:
    use_previous_day = meta["acf_lag_1440"].to_numpy() > DAILY_ACF_MIN
    seasonal = np.empty((len(series), len(starts), HORIZON), dtype=np.float64)
    day_rows = np.flatnonzero(use_previous_day)
    hour_rows = np.flatnonzero(~use_previous_day)
    if len(day_rows):
        seasonal[day_rows] = gather(series[day_rows], starts - MINUTES_PER_DAY, HORIZON)
    if len(hour_rows):
        seasonal[hour_rows] = gather(series[hour_rows], starts - HORIZON, HORIZON)

    last = series[:, starts - 1].astype(np.float64)
    last_value = np.repeat(last[:, :, None], HORIZON, axis=2)

    recent = gather(series, starts - TRAILING_MINUTES, TRAILING_MINUTES).astype(np.float64)
    level = np.quantile(recent, TRAILING_QUANTILE, axis=-1)
    trailing = np.repeat(level[:, :, None], HORIZON, axis=2)
    return {
        "seasonal_naive": seasonal,
        "last_value": last_value,
        "trailing_percentile": trailing,
    }


def check_forecasts(series: np.ndarray, meta: pd.DataFrame, starts: np.ndarray, forecasts: dict[str, np.ndarray], y: np.ndarray) -> None:
    covered = y.reshape(len(y), -1).sum(axis=1)
    expected = np.rint(meta["test_invocations"].to_numpy())
    if not np.array_equal(covered, expected):
        raise SystemExit("Forecast windows do not cover the day 11-14 invocation totals")
    if not np.array_equal(forecasts["last_value"][:, :, 0], series[:, starts - 1]):
        raise SystemExit("Last-value did not repeat the latest minute")
    day_rows = np.flatnonzero(meta["acf_lag_1440"].to_numpy() > DAILY_ACF_MIN)
    if len(day_rows):
        row = int(day_rows[0])
        start = int(starts[0])
        expected_day = series[row, start - MINUTES_PER_DAY : start - MINUTES_PER_DAY + HORIZON]
        if not np.array_equal(forecasts["seasonal_naive"][row, 0], expected_day):
            raise SystemExit("Seasonal naive did not repeat the previous day")


def origin_slices(starts: np.ndarray) -> dict[str, slice]:
    held_out = int(np.searchsorted(starts, HELD_OUT_START))
    return {
        "days_11_14": slice(None),
        "days_11_12": slice(0, held_out),
        "days_13_14": slice(held_out, None),
    }


def main() -> None:
    self_check()
    series, meta = load_series()
    starts = origins()
    y = gather(series, starts, HORIZON).astype(np.float64)
    forecasts = forecast(series, meta, starts)
    check_forecasts(series, meta, starts, forecasts, y)

    use_previous_day = meta["acf_lag_1440"].to_numpy() > DAILY_ACF_MIN
    variant = np.where(use_previous_day, "previous_day", "previous_hour")
    variant_counts = (
        pd.DataFrame({"class": meta["class"], "variant": variant})
        .value_counts()
        .rename("functions")
        .reset_index()
        .sort_values(["class", "variant"])
    )

    per_function = per_function_table(y, forecasts, meta, origin_slices(starts))
    summary = class_table(per_function)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_function.to_csv(OUT_DIR / "baseline_per_function.csv", index=False)
    summary.to_csv(OUT_DIR / "baseline_by_class.csv", index=False)
    np.savez_compressed(
        OUT_DIR / "baseline_predictions.npz",
        y=y,
        yhat=np.stack([forecasts[model] for model in MODELS]),
        model=np.array(MODELS, dtype="U32"),
        origin=starts,
        function_id=meta["function_id"].to_numpy(dtype="U160"),
        label=meta["class"].to_numpy(dtype="U16"),
    )
    protocol = {
        "context_minutes": CONTEXT,
        "horizon_minutes": HORIZON,
        "stride_minutes": STRIDE,
        "origins": int(len(starts)),
        "capacity_invocations_per_container_minute": 10,
        "seasonal_acf_min": DAILY_ACF_MIN,
        "trailing_minutes": TRAILING_MINUTES,
        "trailing_quantile": TRAILING_QUANTILE,
        "windows": list(WINDOWS),
        "seasonal_variant_counts": variant_counts.to_dict(orient="records"),
    }
    (OUT_DIR / "baseline_meta.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print("Seasonal naive variant")
    print(variant_counts.to_string(index=False))
    show = summary[summary["window"] == "days_11_14"][
        [
            "model",
            "class",
            "mae_median",
            "smape_pct_median",
            "under_invocations_median",
            "over_container_minutes_median",
        ]
    ]
    print("\nDays 11-14")
    print(show.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nWrote {OUT_DIR / 'baseline_by_class.csv'}")


if __name__ == "__main__":
    main()
