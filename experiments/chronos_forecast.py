"""Zero-shot Chronos-Bolt forecasts on the same rolling origins.

Uses amazon/chronos-bolt-small. The point forecast is the median. Quantile
forecasts at 0.8 and 0.9 are kept for the provisioning trade-off. The context
is the previous day (1,440 minutes) and the horizon is 60 minutes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

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

MODEL_ID = "amazon/chronos-bolt-small"
QUANTILES = (0.5, 0.8, 0.9)
BATCH_ORIGINS = 16


def load_pipeline():
    from chronos import BaseChronosPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Loading {MODEL_ID} on {device}", flush=True)
    pipeline = BaseChronosPipeline.from_pretrained(
        MODEL_ID,
        device_map=device,
        torch_dtype=dtype,
    )
    return pipeline, device


def predict_batch(pipeline, contexts: np.ndarray) -> dict[float, np.ndarray]:
    """contexts: (batch, CONTEXT) -> quantile arrays (batch, HORIZON)."""
    tensor = torch.tensor(contexts, dtype=torch.float32)
    quantiles, _mean = pipeline.predict_quantiles(
        inputs=tensor,
        prediction_length=HORIZON,
        quantile_levels=list(QUANTILES),
    )
    # quantiles shape: (batch, HORIZON, n_quantiles) or (batch, n_quantiles, HORIZON)
    arr = quantiles.detach().cpu().numpy()
    if arr.shape[-1] == len(QUANTILES):
        return {q: arr[:, :, i] for i, q in enumerate(QUANTILES)}
    if arr.shape[1] == len(QUANTILES):
        return {q: arr[:, i, :] for i, q in enumerate(QUANTILES)}
    raise SystemExit(f"Unexpected Chronos quantile shape: {arr.shape}")


def forecast_all(pipeline, series: np.ndarray, starts: np.ndarray) -> dict[str, np.ndarray]:
    n_func, n_origin = len(series), len(starts)
    out = {f"chronos_q{int(q * 100)}": np.empty((n_func, n_origin, HORIZON), dtype=np.float64) for q in QUANTILES}
    # Keep a clean median name for the point-forecast comparison.
    out["chronos"] = out["chronos_q50"]
    total = n_func * ((n_origin + BATCH_ORIGINS - 1) // BATCH_ORIGINS)
    done = 0
    started = time.perf_counter()
    for i in range(n_func):
        for left in range(0, n_origin, BATCH_ORIGINS):
            right = min(left + BATCH_ORIGINS, n_origin)
            batch_starts = starts[left:right]
            contexts = np.stack(
                [series[i, int(s) - CONTEXT : int(s)].astype(np.float64) for s in batch_starts],
                axis=0,
            )
            preds = predict_batch(pipeline, contexts)
            for q, values in preds.items():
                key = f"chronos_q{int(q * 100)}"
                out[key][i, left:right, :] = np.maximum(values, 0.0)
            done += 1
            if done % 20 == 0 or done == total:
                rate = done / max(time.perf_counter() - started, 1e-6)
                eta = (total - done) / max(rate, 1e-6)
                print(
                    f"Chronos {done}/{total} batches "
                    f"({100 * done / total:.1f}%) eta {eta / 60:.1f} min",
                    flush=True,
                )
    return out


def main() -> None:
    self_check()
    series, meta = load_series()
    starts = origins()
    y = gather(series, starts, HORIZON).astype(np.float64)
    check_coverage(y, meta)
    pipeline, device = load_pipeline()
    forecasts = forecast_all(pipeline, series, starts)

    # Score the point forecast and the two upper quantiles separately.
    score_models = {
        "chronos": forecasts["chronos"],
        "chronos_q80": forecasts["chronos_q80"],
        "chronos_q90": forecasts["chronos_q90"],
    }
    per_function = per_function_table(y, score_models, meta, origin_slices(starts))
    summary = class_table(per_function)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_function.to_csv(OUT_DIR / "chronos_per_function.csv", index=False)
    summary.to_csv(OUT_DIR / "chronos_by_class.csv", index=False)
    np.savez_compressed(
        OUT_DIR / "chronos_predictions.npz",
        y=y,
        chronos=forecasts["chronos"],
        chronos_q80=forecasts["chronos_q80"],
        chronos_q90=forecasts["chronos_q90"],
        origin=starts,
        function_id=meta["function_id"].to_numpy(dtype="U160"),
        label=meta["class"].to_numpy(dtype="U16"),
    )
    (OUT_DIR / "chronos_meta.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "device": device,
                "context_minutes": CONTEXT,
                "horizon_minutes": HORIZON,
                "quantiles": list(QUANTILES),
                "point_forecast": "median (q50)",
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
            "short_container_minutes_median",
            "over_container_minutes_median",
        ]
    ]
    print("\nDays 11-14")
    print(show.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nWrote {OUT_DIR / 'chronos_by_class.csv'}")


if __name__ == "__main__":
    main()
