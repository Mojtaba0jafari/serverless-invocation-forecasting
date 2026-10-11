"""Build the fixed 80-function subset from the Azure Functions 2019 trace.

Class statistics use days 1-10 only. A missing day is treated as 1,440 zeros.
Assignment order is sparse, then daily, then bursty, then other.

  sparse  zero-minute share >= 0.90
  daily   not sparse, and autocorrelation at lag 1,440 > 0.30
  bursty  not sparse or daily, and CV at or above the 75th percentile
          of CV among non-sparse functions
  other   the rest

The sample keeps 20 functions per class. A function is eligible only if it
has at least one invocation in days 1-10 and at least one in days 11-14.
The sample is drawn with seed 42.
"""

from __future__ import annotations

import io
import json
import tarfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "raw" / "azurefunctions-dataset2019.tar.xz"
OUT_DIR = ROOT / "data"

MINUTES_PER_DAY = 1440
TRAIN_DAYS = 10
N_DAYS = 14
TRAIN_MINUTES = TRAIN_DAYS * MINUTES_PER_DAY
ACF_PAIRS = (TRAIN_DAYS - 1) * MINUTES_PER_DAY
SPARSE_ZERO_FRAC = 0.90
DAILY_ACF_MIN = 0.30
BURSTY_QUANTILE = 0.75
PER_CLASS = 20
SEED = 42
ID_COLUMNS = ["HashOwner", "HashApp", "HashFunction", "Trigger"]
CLASSES = ["sparse", "daily", "bursty", "other"]


def iter_invocation_days(archive: Path):
    """Yield (day, dataframe) for invocation files, in day order."""
    with tarfile.open(archive, "r:xz") as tar:
        for member in tar:
            name = Path(member.name).name
            if not name.startswith("invocations_per_function_md.anon.d"):
                continue
            day = int(name.split(".d")[1][:2])
            started = time.perf_counter()
            raw = tar.extractfile(member).read()
            frame = pd.read_csv(io.BytesIO(raw))
            print(
                f"day {day:02d}: {len(frame):,} rows, "
                f"{len(raw) / 1e6:.0f} MB read in {time.perf_counter() - started:.0f}s",
                flush=True,
            )
            del raw
            yield day, frame


def minute_matrix(frame: pd.DataFrame) -> np.ndarray:
    columns = [str(column).strip() for column in frame.columns]
    frame.columns = columns
    if columns[:4] != ID_COLUMNS:
        raise SystemExit(f"Unexpected columns: {columns[:8]}")
    if len(columns) < 4 + MINUTES_PER_DAY:
        raise SystemExit(f"Expected 1,440 minute columns, found {len(columns) - 4}")
    values = frame.iloc[:, 4 : 4 + MINUTES_PER_DAY].to_numpy(dtype=np.float64, copy=False)
    if values.shape[1] != MINUTES_PER_DAY:
        raise SystemExit(f"Minute matrix has shape {values.shape}")
    return values


def accumulate(archive: Path) -> pd.DataFrame:
    sum_v = defaultdict(float)
    sumsq = defaultdict(float)
    nonzero = defaultdict(float)
    sum_x = defaultdict(float)
    sum_y = defaultdict(float)
    sum_xx = defaultdict(float)
    sum_yy = defaultdict(float)
    sum_xy = defaultdict(float)
    test_inv = defaultdict(float)
    owner = {}
    trigger = {}

    prev_keys = None
    prev_sums = None
    prev_sumsq = None
    prev_index = None
    prev_values = None

    for day, frame in iter_invocation_days(archive):
        if frame.duplicated(["HashApp", "HashFunction"]).any():
            raise SystemExit(f"Day {day} has duplicate functions")
        values = minute_matrix(frame)
        apps = frame["HashApp"].astype(str).to_numpy()
        funcs = frame["HashFunction"].astype(str).to_numpy()
        owners = frame["HashOwner"].astype(str).to_numpy()
        triggers = frame["Trigger"].astype(str).to_numpy()
        sums = values.sum(axis=1)
        sumsqs = np.square(values).sum(axis=1)
        nonzeros = np.count_nonzero(values, axis=1)
        keys = list(zip(apps.tolist(), funcs.tolist()))

        if day <= TRAIN_DAYS:
            for i, key in enumerate(keys):
                if key not in owner:
                    owner[key] = owners[i]
                    trigger[key] = triggers[i]
                sum_v[key] += float(sums[i])
                sumsq[key] += float(sumsqs[i])
                nonzero[key] += float(nonzeros[i])
            if day >= 2:
                for j, key in enumerate(prev_keys):
                    sum_x[key] += float(prev_sums[j])
                    sum_xx[key] += float(prev_sumsq[j])
                curr_index = {key: i for i, key in enumerate(keys)}
                common = [key for key in prev_keys if key in curr_index]
                if common:
                    prev_idx = [prev_index[key] for key in common]
                    curr_idx = [curr_index[key] for key in common]
                    dots = np.einsum(
                        "ij,ij->i",
                        prev_values[prev_idx],
                        values[curr_idx],
                    )
                    for key, dot in zip(common, dots):
                        sum_xy[key] += float(dot)
                for i, key in enumerate(keys):
                    sum_y[key] += float(sums[i])
                    sum_yy[key] += float(sumsqs[i])
            prev_keys = keys
            prev_sums = np.array(sums, copy=True)
            prev_sumsq = np.array(sumsqs, copy=True)
            prev_values = np.array(values, copy=True, order="C")
            prev_index = {key: i for i, key in enumerate(keys)}
            if day == TRAIN_DAYS:
                prev_keys = prev_sums = prev_sumsq = prev_values = prev_index = None
        else:
            for i, key in enumerate(keys):
                test_inv[key] += float(sums[i])
        del frame, values

    rows = []
    for key, total in sum_v.items():
        app, func = key
        mean = total / TRAIN_MINUTES
        var = max(sumsq[key] / TRAIN_MINUTES - mean * mean, 0.0)
        std = var**0.5
        cv = std / mean if mean > 0 else np.inf
        num = ACF_PAIRS * sum_xy[key] - sum_x[key] * sum_y[key]
        den_x = ACF_PAIRS * sum_xx[key] - sum_x[key] ** 2
        den_y = ACF_PAIRS * sum_yy[key] - sum_y[key] ** 2
        den = (den_x * den_y) ** 0.5 if den_x > 0 and den_y > 0 else 0.0
        acf = float(num / den) if den > 1e-8 else 0.0
        rows.append(
            {
                "hash_owner": owner[key],
                "hash_app": app,
                "hash_function": func,
                "trigger": trigger[key],
                "zero_frac": 1.0 - nonzero[key] / TRAIN_MINUTES,
                "mean_per_min": mean,
                "std_per_min": std,
                "cv": cv,
                "acf_lag_1440": acf,
                "train_invocations": total,
                "test_invocations": test_inv.get(key, 0.0),
            }
        )
    return pd.DataFrame(rows)


def assign_classes(stats: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    non_sparse = (stats["zero_frac"] < SPARSE_ZERO_FRAC).to_numpy()
    finite_cv = non_sparse & np.isfinite(stats["cv"].to_numpy())
    threshold = float(np.quantile(stats.loc[finite_cv, "cv"], BURSTY_QUANTILE))
    acf = stats["acf_lag_1440"].to_numpy()
    cv = stats["cv"].to_numpy()
    labels = np.full(len(stats), "other", dtype=object)
    labels[non_sparse & (acf > DAILY_ACF_MIN)] = "daily"
    labels[non_sparse & (acf <= DAILY_ACF_MIN) & (cv >= threshold)] = "bursty"
    labels[stats["zero_frac"].to_numpy() >= SPARSE_ZERO_FRAC] = "sparse"
    stats = stats.copy()
    stats["class"] = labels
    return stats, threshold


def sample_functions(stats: pd.DataFrame) -> pd.DataFrame:
    eligible = stats[(stats["train_invocations"] > 0) & (stats["test_invocations"] > 0)]
    rng = np.random.default_rng(SEED)
    chosen = []
    for label in CLASSES:
        pool = eligible[eligible["class"] == label].sort_values(["hash_app", "hash_function"])
        print(f"eligible {label}: {len(pool):,}", flush=True)
        if len(pool) < PER_CLASS:
            raise SystemExit(f"Class {label} has only {len(pool)} eligible functions")
        take = rng.choice(len(pool), size=PER_CLASS, replace=False)
        chosen.append(pool.iloc[np.sort(take)])
    selected = pd.concat(chosen, ignore_index=True)
    selected.insert(0, "function_id", selected["hash_app"] + "/" + selected["hash_function"])
    return selected


def extract_series(archive: Path, selected: pd.DataFrame) -> np.ndarray:
    row_of = {
        (app, func): i
        for i, (app, func) in enumerate(zip(selected["hash_app"], selected["hash_function"]))
    }
    series = np.zeros((len(selected), N_DAYS * MINUTES_PER_DAY), dtype=np.int64)
    for day, frame in iter_invocation_days(archive):
        values = minute_matrix(frame)
        apps = frame["HashApp"].astype(str).to_numpy()
        funcs = frame["HashFunction"].astype(str).to_numpy()
        start = (day - 1) * MINUTES_PER_DAY
        for i, key in enumerate(zip(apps.tolist(), funcs.tolist())):
            row = row_of.get(key)
            if row is not None:
                series[row, start : start + MINUTES_PER_DAY] = np.rint(values[i]).astype(np.int64)
        del frame, values
    return series


def main() -> None:
    if not ARCHIVE.exists():
        raise SystemExit(f"Missing archive: {ARCHIVE}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Pass 1: class statistics", flush=True)
    stats, cv_threshold = assign_classes(accumulate(ARCHIVE))
    stats_path = OUT_DIR / "function_stats.csv"
    stats.to_csv(stats_path, index=False)
    print(f"functions in days 1-10: {len(stats):,}", flush=True)
    print(stats["class"].value_counts().to_string(), flush=True)
    print(f"bursty CV threshold (q{BURSTY_QUANTILE:.2f} of non-sparse): {cv_threshold:.4f}", flush=True)

    print("Sampling", flush=True)
    selected = sample_functions(stats)
    selected_path = OUT_DIR / "functions.csv"
    selected.to_csv(selected_path, index=False)

    print("Pass 2: series for the 80 functions", flush=True)
    series = extract_series(ARCHIVE, selected)
    series_path = OUT_DIR / "selected_series.npz"
    np.savez_compressed(
        series_path,
        series=series,
        function_id=selected["function_id"].to_numpy(dtype="U160"),
        label=selected["class"].to_numpy(dtype="U16"),
    )

    meta = {
        "source": "Azure Functions 2019 invocation traces",
        "archive": ARCHIVE.name,
        "seed": SEED,
        "per_class": PER_CLASS,
        "train_days": [1, TRAIN_DAYS],
        "test_days": [TRAIN_DAYS + 1, N_DAYS],
        "sparse_zero_frac_min": SPARSE_ZERO_FRAC,
        "daily_acf_min": DAILY_ACF_MIN,
        "bursty_cv_quantile": BURSTY_QUANTILE,
        "bursty_cv_threshold": cv_threshold,
        "class_counts": stats["class"].value_counts().to_dict(),
        "selected_counts": selected["class"].value_counts().to_dict(),
        "series_shape": list(series.shape),
    }
    (OUT_DIR / "subset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {selected_path}", flush=True)
    print(f"Wrote {series_path} shape={series.shape}", flush=True)
    print(selected.groupby("class")[["zero_frac", "acf_lag_1440", "cv"]].median().to_string(), flush=True)


if __name__ == "__main__":
    main()
