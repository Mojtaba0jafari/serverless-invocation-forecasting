"""Figures for the revised paper, from thresholded forecasts (results/robust_*)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import HORIZON, OUT_DIR, ROOT, gather, load_series, origins
from robustness import build_forecasts, fix

FIGURES = ROOT / "figures"
DAY = 24


def stitch(block: np.ndarray, day: int) -> np.ndarray:
    flat = block[day * DAY : (day + 1) * DAY].reshape(-1)
    return flat.reshape(-1, 10).mean(axis=1)  # 10-minute means, 144 points


def plot_examples() -> None:
    series, meta = load_series()
    starts = origins()
    y = gather(series, starts, HORIZON).astype(float)
    f = build_forecasts(series, meta, starts)
    f["lstm"] = np.load(OUT_DIR / "lstm_lin_tau0.5_seed42.npz")["yhat"]
    models = [
        ("seasonal_naive", "Seasonal naive", "#0072B2", "-"),
        ("trailing_p95", "Trailing P95", "#E69F00", "-"),
        ("ets", "ETS", "#009E73", "--"),
        ("lstm", "LSTM", "#999999", "--"),
        ("chronos", "Chronos", "#D55E00", ":"),
    ]
    chosen = {"daily": 30, "sparse": 2}
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                         "legend.fontsize": 6, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.3), sharex=True)
    minutes = np.arange(0, 1440, 10)
    for axis, (label, row) in zip(axes, chosen.items()):
        assert meta["class"].iloc[row] == label
        axis.plot(minutes, stitch(y[row], 0), color="black", lw=0.8, label="Actual")
        for key, name, color, style in models:
            axis.plot(minutes, stitch(fix(f[key])[row], 0), color=color, lw=0.8, ls=style, label=name)
        axis.set_xlim(0, 1439)
        axis.set_ylim(bottom=0)
        axis.set_ylabel("Invocations (per min)")
        axis.set_title(f"{label.capitalize()} function, day 11")
    axes[1].set_xlabel("Minute of the day")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(handles, labels, frameon=False, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "forecast_examples.pdf")
    fig.savefig(FIGURES / "forecast_examples.png", dpi=160)
    plt.close(fig)


def plot_frontier() -> None:
    s = pd.read_csv(OUT_DIR / "robust_summary.csv")
    s = s[(s.variant == "C10_thresholded") & (s.window == "days_11_14") & (s["class"] == "all")].set_index("model")
    families = [
        ("Trailing quantile", ["trailing_p50", "trailing_p80", "trailing_p90", "trailing_p95", "trailing_p99"], "#E69F00", "o"),
        ("Chronos quantile", ["chronos", "chronos_q80", "chronos_q90"], "#D55E00", "s"),
        ("Seasonal naive x margin", ["seasonal_naive", "seasonal_x1.25", "seasonal_x1.5", "seasonal_x2.0"], "#0072B2", "^"),
        ("ETS x margin", ["ets", "ets_x1.25", "ets_x1.5"], "#009E73", "D"),
        ("LSTM x margin", ["lstm", "lstm_x1.25", "lstm_x1.5", "lstm_x2.0"], "#888888", "v"),
    ]
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axis = plt.subplots(figsize=(3.45, 2.8))
    for name, keys, color, marker in families:
        axis.plot([s.loc[k, "O_frac"] for k in keys], [s.loc[k, "S_frac"] for k in keys], color=color, marker=marker,
                  ms=3.5, lw=0.9, label=name)
    if "lstm_tau0.9" in s.index:
        axis.scatter([s.loc["lstm_tau0.9", "O_frac"]], [s.loc["lstm_tau0.9", "S_frac"]], marker="v", s=22, facecolor="none",
                     edgecolor="#888888", zorder=4, label="LSTM trained at tau 0.9")
    axis.scatter([s.loc["keepalive10_reactive", "O_frac"]], [s.loc["keepalive10_reactive", "S_frac"]], marker="*", s=60,
                 color="black", zorder=4, label="Keep-alive 10 min (reactive)")
    axis.scatter([s.loc["previous_hour", "O_frac"]], [s.loc["previous_hour", "S_frac"]], marker="x", s=24, color="black",
                 zorder=4, label="Previous hour")
    axis.set_xlim(0, 1.1)
    axis.set_ylim(0, 0.55)
    axis.set_xlabel("Extra capacity (share of required)")
    axis.set_ylabel("Shortfall (share of required)")
    axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "provisioning_tradeoff.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "provisioning_tradeoff.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_examples()
    plot_frontier()
    print("Wrote figures")
