"""Seasonal Holt-Winters forecasts on the same rolling origins.

The model is additive Holt-Winters with a daily season (period 1,440).
Smoothing weights are chosen on days 1-10 only, one triple per function, by
minimizing one-step error on a small grid. The selected weights then filter
the series forward. A forecast at an origin uses only the state built from
minutes before that origin.

A pure daily cycle is the check that this is not the old local Holt model:
the forecast has to follow the cycle instead of a flat level.
"""

from __future__ import annotations

import json
import time

import numpy as np

from common import (
    HORIZON,
    MINUTES_PER_DAY,
    OUT_DIR,
    check_coverage,
    gather,
    load_series,
    origin_slices,
    origins,
)
from metrics import class_table, per_function_table, self_check

MODEL_NAME = "ets"
PERIOD = MINUTES_PER_DAY
ALPHAS = (0.2, 0.4, 0.7)
BETAS = (0.0, 0.05, 0.2)
GAMMAS = (0.05, 0.2, 0.5)


def initialize(history: np.ndarray) -> tuple[float, float, np.ndarray]:
    first = history[:PERIOD]
    second = history[PERIOD : 2 * PERIOD]
    level = float(first.mean())
    trend = float((second.mean() - level) / PERIOD)
    season = first - level
    season = season - season.mean()
    return level, trend, season


def one_step_sse(
    history: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    fit_end: int,
) -> float:
    level, trend, season = initialize(history)
    error = 0.0
    count = 0
    for t in range(fit_end):
        phase = t % PERIOD
        prediction = level + trend + season[phase]
        residual = float(history[t] - prediction)
        if t >= 2 * PERIOD:
            error += residual * residual
            count += 1
        new_level = alpha * (history[t] - season[phase]) + (1.0 - alpha) * (level + trend)
        trend = beta * (new_level - level) + (1.0 - beta) * trend
        season[phase] = gamma * (history[t] - new_level) + (1.0 - gamma) * season[phase]
        level = new_level
    return error / max(count, 1)


def select_weights(history: np.ndarray, fit_end: int) -> tuple[float, float, float]:
    best = (ALPHAS[0], BETAS[0], GAMMAS[0])
    best_error = float("inf")
    for alpha in ALPHAS:
        for beta in BETAS:
            for gamma in GAMMAS:
                error = one_step_sse(history, alpha, beta, gamma, fit_end)
                if error < best_error:
                    best_error = error
                    best = (alpha, beta, gamma)
    return best


def forecast_function(history: np.ndarray, starts: np.ndarray, weights: tuple[float, float, float]) -> np.ndarray:
    alpha, beta, gamma = weights
    level, trend, season = initialize(history)
    out = np.empty((len(starts), HORIZON), dtype=np.float64)
    start_list = [int(value) for value in starts]
    next_origin = 0
    for t in range(start_list[-1] + 1):
        while next_origin < len(start_list) and t == start_list[next_origin]:
            origin = start_list[next_origin]
            horizon = np.empty(HORIZON, dtype=np.float64)
            for step in range(1, HORIZON + 1):
                phase = (origin + step - 1) % PERIOD
                horizon[step - 1] = level + step * trend + season[phase]
            out[next_origin] = np.maximum(horizon, 0.0)
            next_origin += 1
        phase = t % PERIOD
        new_level = alpha * (history[t] - season[phase]) + (1.0 - alpha) * (level + trend)
        trend = beta * (new_level - level) + (1.0 - beta) * trend
        season[phase] = gamma * (history[t] - new_level) + (1.0 - gamma) * season[phase]
        level = new_level
    if next_origin != len(start_list):
        raise SystemExit("Seasonal ETS missed an origin")
    return out


def synthetic_check() -> None:
    """A daily spike train must be forecast from yesterday, not from a flat level."""
    minutes = 14 * PERIOD
    series = np.zeros(minutes, dtype=np.float64)
    series[np.arange(minutes) % PERIOD == 100] = 50.0
    starts = np.array([10 * PERIOD + 60])
    weights = select_weights(series[: 10 * PERIOD], 10 * PERIOD)
    forecast = forecast_function(series, starts, weights)[0]
    # The horizon covers phases 60..119, and the spike sits at phase 100.
    if forecast[40] < 10:
        raise SystemExit(f"Seasonal ETS missed the daily spike: {forecast[40]:.3f}")
    if forecast[0] > 5:
        raise SystemExit(f"Seasonal ETS invented load off the spike: {forecast[0]:.3f}")


def forecast_all(series: np.ndarray, starts: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    fit_end = 10 * PERIOD
    out = np.empty((len(series), len(starts), HORIZON), dtype=np.float64)
    chosen = []
    started = time.perf_counter()
    for i in range(len(series)):
        history = series[i].astype(np.float64)
        if np.nanmax(history[:fit_end]) == np.nanmin(history[:fit_end]):
            level = float(max(history[int(starts[0]) - 1], 0.0))
            out[i] = level
            chosen.append({"alpha": None, "beta": None, "gamma": None, "flat": True})
        else:
            weights = select_weights(history, fit_end)
            out[i] = forecast_function(history, starts, weights)
            chosen.append(
                {"alpha": weights[0], "beta": weights[1], "gamma": weights[2], "flat": False}
            )
        if (i + 1) % 10 == 0 or i + 1 == len(series):
            rate = (i + 1) / max(time.perf_counter() - started, 1e-6)
            print(f"Seasonal ETS {i + 1}/{len(series)} ({rate:.2f} functions/s)", flush=True)
    return out, chosen


def main() -> None:
    self_check()
    synthetic_check()
    series, meta = load_series()
    starts = origins()
    y = gather(series, starts, HORIZON).astype(np.float64)
    check_coverage(y, meta)
    print(
        f"Seasonal Holt-Winters, period {PERIOD}, {len(series)} functions x {len(starts)} origins",
        flush=True,
    )
    yhat, weights = forecast_all(series, starts)
    per_function = per_function_table(y, {MODEL_NAME: yhat}, meta, origin_slices(starts))
    summary = class_table(per_function)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_function.to_csv(OUT_DIR / "ets_per_function.csv", index=False)
    summary.to_csv(OUT_DIR / "ets_by_class.csv", index=False)
    np.savez_compressed(
        OUT_DIR / "ets_predictions.npz",
        y=y,
        yhat=yhat,
        origin=starts,
        function_id=meta["function_id"].to_numpy(dtype="U160"),
        label=meta["class"].to_numpy(dtype="U16"),
    )
    (OUT_DIR / "ets_meta.json").write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "family": "additive Holt-Winters, daily season",
                "period_minutes": PERIOD,
                "horizon_minutes": HORIZON,
                "weights_fit_on": "days 1-10, one-step squared error",
                "alpha_grid": list(ALPHAS),
                "beta_grid": list(BETAS),
                "gamma_grid": list(GAMMAS),
                "weights": weights,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
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
    print(f"\nWrote {OUT_DIR / 'ets_by_class.csv'}")


if __name__ == "__main__":
    main()
