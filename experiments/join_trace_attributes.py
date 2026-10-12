"""Join per-application memory and per-function duration to the 80 functions.

Memory: median over days 1-10 of the app's 50th percentile allocated MB.
Duration: median over days 1-10 of the function's average duration in ms.
Reads the Azure Functions 2019 archive in data/raw. Days 1-10 only.
"""

from __future__ import annotations

import io
import tarfile

import numpy as np
import pandas as pd

from common import ROOT

ARCHIVE = ROOT / "data" / "raw" / "azurefunctions-dataset2019.tar.xz"


def main() -> None:
    fun = pd.read_csv(ROOT / "data" / "functions.csv")
    apps = set(fun["hash_app"].astype(str))
    funcs = set(fun["hash_function"].astype(str))
    mem: dict[str, list[float]] = {}
    dur: dict[tuple[str, str], list[float]] = {}
    with tarfile.open(ARCHIVE, "r:xz") as tar:
        for member in tar:
            name = member.name.split("/")[-1]
            for day in range(1, 11):
                if name == f"app_memory_percentiles.anon.d{day:02d}.csv":
                    df = pd.read_csv(io.BytesIO(tar.extractfile(member).read()), usecols=["HashApp", "AverageAllocatedMb_pct50"])
                    df = df[df["HashApp"].astype(str).isin(apps)]
                    for a, v in zip(df["HashApp"].astype(str), df["AverageAllocatedMb_pct50"]):
                        mem.setdefault(a, []).append(float(v))
                if name == f"function_durations_percentiles.anon.d{day:02d}.csv":
                    df = pd.read_csv(io.BytesIO(tar.extractfile(member).read()), usecols=["HashApp", "HashFunction", "Average"])
                    df = df[df["HashFunction"].astype(str).isin(funcs)]
                    for a, f, v in zip(df["HashApp"].astype(str), df["HashFunction"].astype(str), df["Average"]):
                        dur.setdefault((a, f), []).append(float(v))
            print(name, flush=True) if "d01" in name else None
    fun["memory_mb"] = fun["hash_app"].astype(str).map({a: float(np.median(v)) for a, v in mem.items()})
    keys = list(zip(fun["hash_app"].astype(str), fun["hash_function"].astype(str)))
    fun["avg_duration_ms"] = [float(np.median(dur[k])) if k in dur else np.nan for k in keys]
    fun[["function_id", "class", "memory_mb", "avg_duration_ms"]].to_csv(ROOT / "data" / "function_attributes.csv", index=False)
    print(fun.groupby("class")[["memory_mb", "avg_duration_ms"]].median())
    print("missing memory", int(fun["memory_mb"].isna().sum()), "missing duration", int(fun["avg_duration_ms"].isna().sum()))
    print(fun["avg_duration_ms"].describe())


if __name__ == "__main__":
    main()
