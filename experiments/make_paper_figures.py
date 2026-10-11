"""Write the paper tables and figures from the saved forecast results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
POINT_MODELS = [
    "seasonal_naive",
    "last_value",
    "trailing_percentile",
    "ets",
    "lstm",
    "chronos",
]
LABELS = {
    "seasonal_naive": "Seasonal naive",
    "last_value": "Last value",
    "trailing_percentile": "Trailing percentile",
    "ets": "ETS",
    "lstm": "LSTM",
    "chronos": "Chronos",
    "chronos_q80": "Chronos q80",
    "chronos_q90": "Chronos q90",
    "selection_rule": "Selection rule",
    "always_seasonal_naive": "Always seasonal naive",
    "always_trailing_percentile": "Always trailing percentile",
}
CLASS_ORDER = ["sparse", "daily", "bursty", "other", "all"]
ORIGIN0 = 10 * 1440
DAY_ORIGINS = 1440 // 60


def rounded(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ("smape_pct_median", "under_invocations_median", "over_container_minutes_median"):
        out[column] = out[column].map(lambda value: round(float(value), 1))
    return out


def write_tables() -> None:
    summary = pd.read_csv(RESULTS / "all_by_class.csv")
    table1 = summary[
        (summary["window"] == "days_11_14") & (summary["model"].isin(POINT_MODELS))
    ][
        [
            "model",
            "class",
            "smape_pct_median",
            "under_invocations_median",
            "over_container_minutes_median",
        ]
    ].copy()
    table1["class"] = pd.Categorical(table1["class"], CLASS_ORDER, ordered=True)
    table1["model"] = pd.Categorical(table1["model"], POINT_MODELS, ordered=True)
    table1 = rounded(table1.sort_values(["model", "class"]))
    table1.to_csv(RESULTS / "table1_days_11_14.csv", index=False)

    held = pd.read_csv(RESULTS / "selection_rule_score.csv")
    table2 = held[
        (held["window"] == "days_13_14")
        & (held["model"].isin(["selection_rule", "always_seasonal_naive", "always_trailing_percentile"]))
    ][
        [
            "model",
            "class",
            "smape_pct_median",
            "under_invocations_median",
            "over_container_minutes_median",
        ]
    ].copy()
    table2["class"] = pd.Categorical(table2["class"], CLASS_ORDER, ordered=True)
    table2 = rounded(table2.sort_values(["model", "class"]))
    table2.to_csv(RESULTS / "table2_days_13_14.csv", index=False)
    print(table1.to_string(index=False))
    print(table2.to_string(index=False))


def stitch(block: np.ndarray, day: int) -> np.ndarray:
    start = day * DAY_ORIGINS
    return block[start : start + DAY_ORIGINS].reshape(-1)


def choose_examples(y: np.ndarray, labels: np.ndarray, ids: np.ndarray) -> dict[str, tuple[int, int]]:
    """Fixed rows chosen because the day is visible, not because a model wins there."""
    del y, ids
    chosen = {"daily": (30, 0), "sparse": (2, 0)}
    for label, (row, day) in chosen.items():
        if labels[row] != label:
            raise SystemExit(f"Row {row} is {labels[row]}, expected {label}")
        print(label, "row", row, "day", day + 11)
    return chosen


def plot_examples() -> None:
    bundle = np.load(RESULTS / "baseline_predictions.npz")
    y = bundle["y"]
    yhat = bundle["yhat"]
    model_names = [str(name) for name in bundle["model"]]
    seasonal = yhat[model_names.index("seasonal_naive")]
    trailing = yhat[model_names.index("trailing_percentile")]
    ids = bundle["function_id"].astype(str)
    labels = bundle["label"].astype(str)
    chosen = choose_examples(y, labels, ids)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.15), sharex=True)
    for axis, label in zip(axes, ("daily", "sparse")):
        row, day = chosen[label]
        minutes = np.arange(1440)
        actual = stitch(y[row], day)
        axis.plot(minutes, actual, color="black", lw=0.8, label="Actual")
        axis.plot(minutes, stitch(seasonal[row], day), color="#0072B2", lw=0.9, label="Seasonal naive")
        axis.plot(minutes, stitch(trailing[row], day), color="#E69F00", lw=0.9, label="Trailing percentile")
        axis.set_xlim(0, 1439)
        axis.set_ylim(bottom=0)
        axis.set_ylabel("Invocations")
        axis.set_title(f"{label.capitalize()} function, day {day + 11}")
    axes[1].set_xlabel("Minute of the day")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(handles, labels, frameon=False, loc="upper right", fontsize=6.5)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "forecast_examples.pdf")
    fig.savefig(FIGURES / "forecast_examples.png", dpi=160)
    plt.close(fig)


def plot_tradeoff() -> None:
    summary = pd.read_csv(RESULTS / "all_by_class.csv")
    models = POINT_MODELS + ["chronos_q80", "chronos_q90"]
    part = summary[
        (summary["window"] == "days_11_14")
        & (summary["class"] == "all")
        & (summary["model"].isin(models))
    ].set_index("model")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axis = plt.subplots(figsize=(3.45, 2.7))
    for model in models:
        row = part.loc[model]
        x = float(row["over_container_minutes_median"])
        y_value = float(row["under_invocations_median"])
        quantile = model.startswith("chronos_q")
        axis.scatter(
            x,
            y_value,
            s=28,
            color="#D55E00" if quantile else "#0072B2",
            marker="s" if quantile else "o",
            zorder=3,
            label=LABELS[model],
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(1.5, 20000)
    axis.set_ylim(50, 4000)
    axis.set_xlabel("Median extra container-minutes")
    axis.set_ylabel("Median missed invocations")
    axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "provisioning_tradeoff.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "provisioning_tradeoff.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    write_tables()
    plot_examples()
    plot_tradeoff()
    print("Wrote figures")
