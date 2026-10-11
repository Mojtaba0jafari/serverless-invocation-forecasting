"""Score invocation forecasts with error and provisioning cost.

Each test minute is scored once. Origins are 60 minutes apart and the horizon
is 60 minutes, so the windows over days 11-14 do not overlap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CAPACITY = 10
WINDOWS = ("days_11_14", "days_11_12", "days_13_14")
CLASSES = ("sparse", "daily", "bursty", "other", "all")


def self_check() -> None:
    """Match the worked example: load 100, capacity 10."""
    y = np.array([[[100.0]]])
    short = score_arrays(y, np.array([[[80.0]]]))
    extra = score_arrays(y, np.array([[[130.0]]]))
    # Forecast 80 prepares 8 containers against 10 needed.
    if short["short_container_minutes"][0] != 2 or short["over_container_minutes"][0] != 0:
        raise SystemExit("Provisioning check failed for a forecast of 80")
    # Forecast 130 prepares 13 containers: 3 idle container-minutes, no shortfall.
    if extra["short_container_minutes"][0] != 0 or extra["over_container_minutes"][0] != 3:
        raise SystemExit("Provisioning check failed for a forecast of 130")
    # 8 and 5 both fit in one container, so the forecast error is not a shortfall.
    same_container = score_arrays(np.array([[[8.0]]]), np.array([[[5.0]]]))
    if same_container["short_container_minutes"][0] != 0 or same_container["over_container_minutes"][0] != 0:
        raise SystemExit("Provisioning check failed for a forecast inside the same container")
    zeros = score_arrays(np.array([[[0.0]]]), np.array([[[0.0]]]))
    if zeros["smape_pct"][0] != 0:
        raise SystemExit("sMAPE of a zero minute should be 0")


def score_arrays(y: np.ndarray, yhat: np.ndarray, capacity: float = CAPACITY) -> dict[str, np.ndarray]:
    """Score arrays shaped (function, origin, minute).

    Forecasts are clipped at zero before the container count, so a negative
    prediction prepares no containers.
    """
    if y.shape != yhat.shape:
        raise ValueError(f"Shape mismatch: {y.shape} vs {yhat.shape}")
    if capacity <= 0:
        raise ValueError("Capacity must be positive")
    yhat = np.maximum(yhat, 0)
    error = np.abs(y - yhat)
    denom = np.abs(y) + np.abs(yhat)
    smape_term = np.divide(2 * error, denom, out=np.zeros_like(error), where=denom > 0)
    positive = y > 0
    positive_count = positive.sum(axis=(1, 2))
    positive_error = np.where(positive, error, 0).sum(axis=(1, 2))
    return {
        "mae": error.mean(axis=(1, 2)),
        "smape_pct": 100 * smape_term.mean(axis=(1, 2)),
        "mae_nonzero": np.divide(
            positive_error,
            positive_count,
            out=np.full(len(y), np.nan),
            where=positive_count > 0,
        ),
        "short_container_minutes": np.maximum(
            np.ceil(y / capacity) - np.ceil(yhat / capacity),
            0,
        ).sum(axis=(1, 2)),
        "over_container_minutes": np.maximum(
            np.ceil(yhat / capacity) - np.ceil(y / capacity),
            0,
        ).sum(axis=(1, 2)),
    }


def per_function_table(
    y: np.ndarray,
    forecasts: dict[str, np.ndarray],
    meta: pd.DataFrame,
    origin_slices: dict[str, slice],
) -> pd.DataFrame:
    frames = []
    for window, origin_slice in origin_slices.items():
        actual = y[:, origin_slice, :]
        for model, yhat in forecasts.items():
            scored = score_arrays(actual, yhat[:, origin_slice, :])
            frame = pd.DataFrame(scored)
            frame.insert(0, "function_id", meta["function_id"].to_numpy())
            frame.insert(1, "class", meta["class"].to_numpy())
            frame.insert(2, "model", model)
            frame.insert(3, "window", window)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def class_table(per_function: pd.DataFrame) -> pd.DataFrame:
    overall = per_function.copy()
    overall["class"] = "all"
    both = pd.concat([per_function, overall], ignore_index=True)
    grouped = both.groupby(["window", "model", "class"], sort=False)
    summary = grouped.agg(
        mae=("mae", "mean"),
        mae_median=("mae", "median"),
        smape_pct=("smape_pct", "mean"),
        smape_pct_median=("smape_pct", "median"),
        mae_nonzero=("mae_nonzero", "mean"),
        short_container_minutes_sum=("short_container_minutes", "sum"),
        over_container_minutes_sum=("over_container_minutes", "sum"),
        short_container_minutes_median=("short_container_minutes", "median"),
        over_container_minutes_median=("over_container_minutes", "median"),
        n_functions=("function_id", "size"),
    ).reset_index()
    summary["window"] = pd.Categorical(summary["window"], WINDOWS, ordered=True)
    summary["class"] = pd.Categorical(summary["class"], CLASSES, ordered=True)
    return summary.sort_values(["window", "model", "class"]).reset_index(drop=True)
