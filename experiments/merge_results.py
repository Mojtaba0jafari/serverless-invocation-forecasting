"""Merge baseline, ETS, Chronos, and LSTM tables and pick the class-aware rule."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common import OUT_DIR

POINT_MODELS = [
    "seasonal_naive",
    "last_value",
    "trailing_percentile",
    "ets",
    "chronos",
    "lstm",
]


def load_parts() -> pd.DataFrame:
    frames = []
    for name in (
        "baseline_per_function.csv",
        "ets_per_function.csv",
        "chronos_per_function.csv",
        "lstm_per_function.csv",
    ):
        path = OUT_DIR / name
        if not path.exists():
            print(f"missing {path}")
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit("No per-function result files found")
    return pd.concat(frames, ignore_index=True)


def class_summary(per_function: pd.DataFrame) -> pd.DataFrame:
    from metrics import class_table

    return class_table(per_function)


def choose_rule(per_function: pd.DataFrame) -> pd.DataFrame:
    """On days 11-12, pick a point model per class.

    Primary key: median short container-minutes.
    Tie breakers: median over-provisioned container-minutes, then median sMAPE.
    Only the median (point) Chronos forecast competes, not the upper quantiles.
    """
    sel = per_function[
        (per_function["window"] == "days_11_12") & (per_function["model"].isin(POINT_MODELS))
    ]
    rows = []
    for label, group in sel.groupby("class"):
        ranked = (
            group.groupby("model", sort=False)
            .agg(
                under_median=("short_container_minutes", "median"),
                over_median=("over_container_minutes", "median"),
                smape_median=("smape_pct", "median"),
            )
            .reset_index()
            .sort_values(["under_median", "over_median", "smape_median"])
        )
        best = ranked.iloc[0]
        rows.append(
            {
                "class": label,
                "chosen_model": best["model"],
                "under_median_days_11_12": best["under_median"],
                "over_median_days_11_12": best["over_median"],
                "smape_median_days_11_12": best["smape_median"],
            }
        )
    return pd.DataFrame(rows)


def score_rule(per_function: pd.DataFrame, rule: pd.DataFrame) -> pd.DataFrame:
    held = per_function[
        (per_function["window"] == "days_13_14") & (per_function["model"].isin(POINT_MODELS))
    ]
    mapping = dict(zip(rule["class"], rule["chosen_model"]))
    chosen_rows = []
    for label, model in mapping.items():
        part = held[(held["class"] == label) & (held["model"] == model)].copy()
        part["model"] = "selection_rule"
        chosen_rows.append(part)
    always = []
    for model in ("chronos", "trailing_percentile", "seasonal_naive"):
        part = held[held["model"] == model].copy()
        part["model"] = f"always_{model}"
        always.append(part)
    combined = pd.concat(chosen_rows + always, ignore_index=True)
    from metrics import class_table

    return class_table(combined)


def main() -> None:
    per_function = load_parts()
    summary = class_summary(per_function)
    rule = choose_rule(per_function)
    rule_score = score_rule(per_function, rule)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_function.to_csv(OUT_DIR / "all_per_function.csv", index=False)
    summary.to_csv(OUT_DIR / "all_by_class.csv", index=False)
    rule.to_csv(OUT_DIR / "selection_rule.csv", index=False)
    rule_score.to_csv(OUT_DIR / "selection_rule_score.csv", index=False)
    (OUT_DIR / "selection_rule.json").write_text(
        rule.to_json(orient="records", indent=2),
        encoding="utf-8",
    )

    point = summary[
        (summary["window"] == "days_11_14") & (summary["model"].isin(POINT_MODELS + ["chronos_q80", "chronos_q90"]))
    ][
        [
            "model",
            "class",
            "mae_median",
            "smape_pct_median",
            "short_container_minutes_median",
            "over_container_minutes_median",
        ]
    ]
    print("Days 11-14 medians")
    print(point.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nSelection rule (chosen on days 11-12)")
    print(rule.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nHeld-out days 13-14")
    show = rule_score[rule_score["window"] == "days_13_14"][
        [
            "model",
            "class",
            "short_container_minutes_median",
            "over_container_minutes_median",
            "smape_pct_median",
        ]
    ]
    print(show.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
