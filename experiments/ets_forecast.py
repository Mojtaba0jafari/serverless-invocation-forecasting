"""ETS forecasts on the same rolling origins as the baselines.

Model: damped additive Holt (ETS with trend, no seasonal component), fitted
on the previous 360 minutes at each origin. Seasonality is left to the
seasonal-naive baseline and to Chronos; this model captures local level and
trend under the same context length used by the other methods. Failed fits
fall back to the last observed minute.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from common import (
    CONTEXT,
    HORIZON,
    OUT_DIR,
    check_coverage,
    gather,
    load_series,
    origin_slices,
    origins,
)
from metrics import class_table, per_function_table, self_check

MODEL_NAME = "ets"


def fit_one(history: np.ndarray) -> np.ndarray:
    history = np.asarray(history, dtype=np.float64)
    fallback = float(max(history[-1], 0.0))
    if len(history) < 10 or not np.isfinite(history).all():
        return np.full(HORIZON, fallback, dtype=np.float64)
    if np.nanmax(history) == np.nanmin(history):
        return np.full(HORIZON, fallback, dtype=np.float64)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = ExponentialSmoothing(
                history,
                trend="add",
                damped_trend=True,
                seasonal=None,
                initialization_method="estimated",
            ).fit(optimized=True, remove_bias=False)
            pred = np.asarray(fitted.forecast(HORIZON), dtype=np.float64)
            if not np.isfinite(pred).all():
                return np.full(HORIZON, fallback, dtype=np.float64)
            return np.maximum(pred, 0.0)
    except Exception:
        return np.full(HORIZON, fallback, dtype=np.float64)


def forecast_all(series: np.ndarray, starts: np.ndarray) -> np.ndarray:
    out = np.empty((len(series), len(starts), HORIZON), dtype=np.float64)
    total = len(series) * len(starts)
    done = 0
    started = time.perf_counter()
    for i in range(len(series)):
        for j, start in enumerate(starts):
            begin = int(start) - CONTEXT
            out[i, j] = fit_one(series[i, begin:int(start)])
            done += 1
            if done % 400 == 0 or done == total:
                rate = done / max(time.perf_counter() - started, 1e-6)
                eta = (total - done) / max(rate, 1e-6)
                print(
                    f"ETS {done}/{total} ({100 * done / total:.1f}%) "
                    f"{rate:.1f} fits/s eta {eta / 60:.1f} min",
                    flush=True,
                )
    return out


def main() -> None:
    self_check()
    series, meta = load_series()
    starts = origins()
    y = gather(series, starts, HORIZON).astype(np.float64)
    check_coverage(y, meta)
    print(
        f"Fitting damped Holt on {CONTEXT}-minute contexts: "
        f"{len(series)} functions x {len(starts)} origins",
        flush=True,
    )
    yhat = forecast_all(series, starts)
    forecasts = {MODEL_NAME: yhat}
    per_function = per_function_table(y, forecasts, meta, origin_slices(starts))
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
                "family": "damped additive Holt / ETS(A,Ad,N)",
                "context_minutes": CONTEXT,
                "horizon_minutes": HORIZON,
                "note": "No seasonal component; compare with seasonal_naive for day-lag patterns.",
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
