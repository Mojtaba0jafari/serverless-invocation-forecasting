"""LSTM with a linear output head, and an asymmetric-loss variant.

The first LSTM used a ReLU on the output layer. Some of the 60 output units
died and predicted exactly 0 for every window (70% of minutes on daily
functions), which made the LSTM look much worse than the model class is. This
script replaces the ReLU head with a linear head (forecasts are still clipped
at 0 afterwards) and trains with the pinball loss at level tau. tau = 0.5 is
the original absolute-error loss up to a factor 2. tau = 0.9 penalises
under-prediction nine times more than over-prediction.

Usage: python lstm_variants.py TAU SEED
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch
from torch import nn

import lstm_forecast as base
from common import HORIZON, OUT_DIR, check_coverage, gather, load_series, origins

torch.set_num_threads(4)
base.MAX_EPOCHS = 12


class LinearForecaster(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=base.HIDDEN, num_layers=1, batch_first=True)
        self.head = nn.Linear(base.HIDDEN, HORIZON)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        output, _state = self.lstm(context.unsqueeze(-1))
        return self.head(output[:, -1, :])


def make_run_epoch(tau: float):
    def run_epoch(model, contexts, targets, scales, order, optimizer):
        total = 0.0
        count = 0
        for left in range(0, len(order), base.BATCH):
            batch = order[left : left + base.BATCH]
            x = torch.tensor(contexts[batch])
            y_norm = torch.tensor(targets[batch])
            scale = torch.tensor(scales[batch])
            if optimizer is not None:
                optimizer.zero_grad()
            pred = model(x)
            diff = y_norm - pred
            loss = torch.mean(torch.maximum(tau * diff, (tau - 1.0) * diff))
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            d = (y_norm - torch.clamp(pred, min=0.0)) * scale
            total += torch.maximum(tau * d, (tau - 1.0) * d).sum().item()
            count += pred.numel()
        return total / count

    return run_epoch


def main() -> None:
    tau, seed = float(sys.argv[1]), int(sys.argv[2])
    series, meta = load_series()
    starts = origins()
    y = gather(series, starts, HORIZON).astype(np.float64)
    check_coverage(y, meta)
    base.SEED = seed
    base.Forecaster = LinearForecaster
    base.run_epoch = make_run_epoch(tau)
    model, info = base.train(series)
    yhat = base.predict(model, series, starts)
    key = f"lstm_lin_tau{tau}_seed{seed}"
    np.savez_compressed(OUT_DIR / f"{key}.npz", yhat=yhat)
    (OUT_DIR / f"{key}.json").write_text(
        json.dumps({"tau": tau, "seed": seed, "best_val_loss": info["best_val_mae"], "epochs": info["epochs_ran"], "zero_share": float((yhat == 0).mean())}, indent=2),
        encoding="utf-8",
    )
    print(key, info["best_val_mae"], info["epochs_ran"], "zero share", float((yhat == 0).mean()), flush=True)


if __name__ == "__main__":
    main()
