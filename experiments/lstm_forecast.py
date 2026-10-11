"""One shared LSTM forecaster on the same rolling origins as the other models.

The network is trained only on days 1-8 and early-stopped on days 9-10.
Days 11-14 are used only for the reported forecasts. Each window is the
previous day (1,440 minutes), scaled by the mean of that context. The
60-minute prediction is scaled back. Seed is 42.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch
from torch import nn

from common import (
    CONTEXT,
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

MODEL_NAME = "lstm"
SEED = 42
HIDDEN = 32
MAX_EPOCHS = 20
PATIENCE = 3
LEARNING_RATE = 1e-3
BATCH = 128
TRAIN_END = 8 * MINUTES_PER_DAY
VAL_END = 10 * MINUTES_PER_DAY


class Forecaster(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=HIDDEN, num_layers=1, batch_first=True)
        self.head = nn.Linear(HIDDEN, HORIZON)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        output, _state = self.lstm(context.unsqueeze(-1))
        return torch.relu(self.head(output[:, -1, :]))


def window_starts(first: int, last_inclusive: int) -> np.ndarray:
    return np.arange(first, last_inclusive + 1, 60)


def build_windows(series: np.ndarray, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    contexts = np.empty((len(series) * len(starts), CONTEXT), dtype=np.float32)
    targets = np.empty((len(series) * len(starts), HORIZON), dtype=np.float32)
    scales = np.empty((len(series) * len(starts), 1), dtype=np.float32)
    row = 0
    for function in series:
        for start in starts:
            context = function[int(start) - CONTEXT : int(start)]
            scale = max(float(context.mean()), 1.0)
            contexts[row] = context / scale
            targets[row] = function[int(start) : int(start) + HORIZON] / scale
            scales[row, 0] = scale
            row += 1
    return contexts, targets, scales


def run_epoch(
    model: Forecaster,
    contexts: np.ndarray,
    targets: np.ndarray,
    scales: np.ndarray,
    order: np.ndarray,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    total_abs = 0.0
    total_count = 0
    for left in range(0, len(order), BATCH):
        batch = order[left : left + BATCH]
        x = torch.tensor(contexts[batch])
        y_norm = torch.tensor(targets[batch])
        scale = torch.tensor(scales[batch])
        if optimizer is not None:
            optimizer.zero_grad()
        pred_norm = model(x)
        loss = torch.mean(torch.abs(pred_norm - y_norm))
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        original_error = torch.abs(pred_norm * scale - y_norm * scale).sum().item()
        total_abs += original_error
        total_count += pred_norm.numel()
    return total_abs / total_count


def train(series: np.ndarray) -> tuple[Forecaster, dict]:
    torch.manual_seed(SEED)
    train_starts = window_starts(CONTEXT, TRAIN_END - HORIZON)
    val_starts = window_starts(TRAIN_END, VAL_END - HORIZON)
    print(f"Train windows {len(series) * len(train_starts):,}; val windows {len(series) * len(val_starts):,}", flush=True)
    train_x, train_y, train_scale = build_windows(series, train_starts)
    val_x, val_y, val_scale = build_windows(series, val_starts)
    model = Forecaster()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    generator = np.random.default_rng(SEED)
    best_state = None
    best_val = float("inf")
    stale = 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        started = time.perf_counter()
        model.train()
        order = generator.permutation(len(train_x))
        train_mae = run_epoch(model, train_x, train_y, train_scale, order, optimizer)
        model.eval()
        with torch.no_grad():
            val_mae = run_epoch(model, val_x, val_y, val_scale, np.arange(len(val_x)), None)
        history.append({"epoch": epoch, "train_mae": train_mae, "val_mae": val_mae})
        print(
            f"epoch {epoch}: train MAE {train_mae:.3f}  val MAE {val_mae:.3f}  "
            f"({time.perf_counter() - started:.0f}s)",
            flush=True,
        )
        if val_mae < best_val - 1e-4:
            best_val = val_mae
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"Early stop at epoch {epoch}; best val MAE {best_val:.3f}", flush=True)
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_val_mae": best_val, "epochs_ran": history[-1]["epoch"], "history": history}


def predict(model: Forecaster, series: np.ndarray, starts: np.ndarray) -> np.ndarray:
    forecasts = np.empty((len(series), len(starts), HORIZON), dtype=np.float64)
    with torch.no_grad():
        for i, function in enumerate(series):
            contexts = np.stack([function[int(s) - CONTEXT : int(s)] for s in starts]).astype(np.float32)
            scales = np.maximum(contexts.mean(axis=1, keepdims=True), 1.0).astype(np.float32)
            pred = model(torch.tensor(contexts / scales)).numpy() * scales
            forecasts[i] = np.maximum(pred, 0.0)
    return forecasts


def main() -> None:
    self_check()
    series, meta = load_series()
    starts = origins()
    y = gather(series, starts, HORIZON).astype(np.float64)
    check_coverage(y, meta)
    model, train_info = train(series)
    yhat = predict(model, series, starts)
    per_function = per_function_table(y, {MODEL_NAME: yhat}, meta, origin_slices(starts))
    summary = class_table(per_function)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_function.to_csv(OUT_DIR / "lstm_per_function.csv", index=False)
    summary.to_csv(OUT_DIR / "lstm_by_class.csv", index=False)
    np.savez_compressed(
        OUT_DIR / "lstm_predictions.npz",
        y=y,
        yhat=yhat,
        origin=starts,
        function_id=meta["function_id"].to_numpy(dtype="U160"),
        label=meta["class"].to_numpy(dtype="U16"),
    )
    (OUT_DIR / "lstm_meta.json").write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "hidden": HIDDEN,
                "layers": 1,
                "seed": SEED,
                "context_minutes": CONTEXT,
                "horizon_minutes": HORIZON,
                "train_days": [1, 8],
                "early_stop_days": [9, 10],
                "best_val_mae": train_info["best_val_mae"],
                "epochs_ran": train_info["epochs_ran"],
                "history": train_info["history"],
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
    print(f"\nWrote {OUT_DIR / 'lstm_by_class.csv'}")


if __name__ == "__main__":
    main()
