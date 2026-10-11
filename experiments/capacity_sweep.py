"""Rescore saved forecasts at several container capacities.

The forecasts do not change. Only the ceiling that turns an invocation count
into a container count changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import OUT_DIR
from metrics import self_check, score_arrays

CAPACITIES = (1, 5, 10, 20)
MODELS = (
    "seasonal_naive",
    "last_value",
    "trailing_percentile",
    "ets",
    "lstm",
    "chronos",
    "chronos_q80",
    "chronos_q90",
)


def load_forecasts() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    baseline = np.load(OUT_DIR / "baseline_predictions.npz")
    y = baseline["y"]
    names = [str(name) for name in baseline["model"]]
    forecasts = {name: baseline["yhat"][index] for index, name in enumerate(names)}
    ets = np.load(OUT_DIR / "ets_predictions.npz")
    lstm = np.load(OUT_DIR / "lstm_predictions.npz")
    chronos = np.load(OUT_DIR / "chronos_predictions.npz")
    if not (np.array_equal(y, ets["y"]) and np.array_equal(y, lstm["y"]) and np.array_equal(y, chronos["y"])):
        raise SystemExit("Saved forecasts do not share the same actual series")
    forecasts["ets"] = ets["yhat"]
    forecasts["lstm"] = lstm["yhat"]
    forecasts["chronos"] = chronos["chronos"]
    forecasts["chronos_q80"] = chronos["chronos_q80"]
    forecasts["chronos_q90"] = chronos["chronos_q90"]
    return y, forecasts


def main() -> None:
    self_check()
    y, forecasts = load_forecasts()
    rows = []
    for capacity in CAPACITIES:
        for model in MODELS:
            scored = score_arrays(y, forecasts[model], capacity=capacity)
            short = scored["short_container_minutes"]
            extra = scored["over_container_minutes"]
            rows.append(
                {
                    "capacity": capacity,
                    "model": model,
                    "short_median": float(np.median(short)),
                    "extra_median": float(np.median(extra)),
                    "functions_short_pct": float(100 * np.mean(short > 0)),
                }
            )
    table = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "capacity_sweep.csv", index=False)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.1f}"))


if __name__ == "__main__":
    main()
