"""Rescore saved forecasts with container shortfall.

The prediction archives stay as they are. This rewrites the per-function and
per-class tables from those archives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import OUT_DIR, origin_slices
from metrics import class_table, per_function_table, self_check


def meta_from(bundle: np.lib.npyio.NpzFile) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "function_id": bundle["function_id"].astype(str),
            "class": bundle["label"].astype(str),
        }
    )


def write_pair(name: str, per_function: pd.DataFrame) -> None:
    summary = class_table(per_function)
    per_function.to_csv(OUT_DIR / f"{name}_per_function.csv", index=False)
    summary.to_csv(OUT_DIR / f"{name}_by_class.csv", index=False)
    show = summary[summary["window"] == "days_11_14"][
        [
            "model",
            "class",
            "smape_pct_median",
            "short_container_minutes_median",
            "over_container_minutes_median",
        ]
    ]
    print(f"\n{name}, days 11-14")
    print(show.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


def main() -> None:
    self_check()
    baseline = np.load(OUT_DIR / "baseline_predictions.npz")
    starts = baseline["origin"]
    slices = origin_slices(starts)
    meta = meta_from(baseline)
    y = baseline["y"]
    model_names = [str(name) for name in baseline["model"]]
    forecasts = {name: baseline["yhat"][index] for index, name in enumerate(model_names)}
    write_pair("baseline", per_function_table(y, forecasts, meta, slices))

    ets = np.load(OUT_DIR / "ets_predictions.npz")
    write_pair("ets", per_function_table(ets["y"], {"ets": ets["yhat"]}, meta_from(ets), slices))

    lstm = np.load(OUT_DIR / "lstm_predictions.npz")
    write_pair("lstm", per_function_table(lstm["y"], {"lstm": lstm["yhat"]}, meta_from(lstm), slices))

    chronos = np.load(OUT_DIR / "chronos_predictions.npz")
    chronos_forecasts = {
        "chronos": chronos["chronos"],
        "chronos_q80": chronos["chronos_q80"],
        "chronos_q90": chronos["chronos_q90"],
    }
    write_pair(
        "chronos",
        per_function_table(chronos["y"], chronos_forecasts, meta_from(chronos), slices),
    )


if __name__ == "__main__":
    main()
