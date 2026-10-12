"""Scoring-only robustness analysis for the review round.

No model is refit here except two cheap intermittent-demand forecasters.
Everything reads saved predictions.

Changes against the first scoring:
1. Forecasts below 0.5 invocations are set to 0 for every model. A continuous
   forecast such as 0.01 otherwise prepares a whole container (ceiling), and
   that artifact, not hedging, drove most of the idle minutes of Chronos/ETS.
2. A headroom family for every model, so models are compared at equal shortfall.
3. A new-container (cold-start event) proxy, per-function capacity from the
   average execution duration, and the original fixed capacity of 10.
4. Class-reweighted numbers, tails, normalised shortfall, bootstrap CIs.
5. Keep-alive, Croston, TSB, ungated seasonal naive baselines.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import HORIZON, OUT_DIR, ROOT, gather, load_series, origin_slices, origins

CAP = 10.0
THRESH = 0.5
CLASS_N = {"sparse": 37155, "daily": 7673, "bursty": 1809, "other": 3882}  # eligible functions per class
CLASSES = ["sparse", "daily", "bursty", "other"]
B = 2000


def fix(yhat: np.ndarray) -> np.ndarray:
    return np.where(yhat < THRESH, 0.0, yhat)


def score(y, r_prev, yhat=None, p=None, cap=CAP, memory=None):
    """Return per-function dict. y: (F,O,M). cap scalar or (F,1,1)."""
    r = np.ceil(y / cap)
    if p is None:
        p = np.ceil(np.maximum(yhat, 0) / cap)
        err = np.abs(y - np.maximum(yhat, 0))
        den = y + np.maximum(yhat, 0)
        sm = np.divide(2 * err, den, out=np.zeros_like(err), where=den > 0)
        smape = 100 * sm.mean(axis=(1, 2))
        mae = err.mean(axis=(1, 2))
    else:
        smape = np.full(len(y), np.nan)
        mae = np.full(len(y), np.nan)
    short = np.maximum(r - p, 0).sum(axis=(1, 2))
    over = np.maximum(p - r, 0).sum(axis=(1, 2))
    cold = np.maximum(r - np.maximum(p, r_prev), 0).sum(axis=(1, 2))
    return {"smape": smape, "mae": mae, "S": short, "O": over, "cold": cold, "req": r.sum(axis=(1, 2))}


def wmedian(values, weights):
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w) / w.sum()
    return float(v[np.searchsorted(cum, 0.5)])


def tsb(ctx, alpha=0.1, beta=0.1):
    """ctx (N, T). Teunter-Syntetos-Babai forecast level per row."""
    nz0 = ctx > 0
    p = nz0.mean(axis=1)
    z = np.where(nz0.any(axis=1), (ctx * nz0).sum(axis=1) / np.maximum(nz0.sum(axis=1), 1), 0.0)
    for t in range(ctx.shape[1]):
        x = ctx[:, t]
        nz = x > 0
        p = p + beta * (nz - p)
        z = np.where(nz, z + alpha * (x - z), z)
    return p * z


def croston(ctx, alpha=0.1):
    nz0 = ctx > 0
    n = ctx.shape[1]
    z = np.where(nz0.any(axis=1), (ctx * nz0).sum(axis=1) / np.maximum(nz0.sum(axis=1), 1), 0.0)
    q = np.where(nz0.any(axis=1), n / np.maximum(nz0.sum(axis=1), 1), np.inf)
    gap = np.ones(len(ctx))
    for t in range(n):
        x = ctx[:, t]
        nz = x > 0
        z = np.where(nz, z + alpha * (x - z), z)
        q = np.where(nz & np.isfinite(q), q + alpha * (gap - q), q)
        gap = np.where(nz, 1.0, gap + 1.0)
    return np.where(np.isfinite(q), z / q, 0.0)


def build_forecasts(series, meta, starts):
    base = np.load(OUT_DIR / "baseline_predictions.npz")
    f = {name: base["yhat"][i] for i, name in enumerate(base["model"])}
    f["ets"] = np.load(OUT_DIR / "ets_predictions.npz")["yhat"]
    f["lstm"] = np.load(OUT_DIR / "lstm_predictions.npz")["yhat"]
    ch = np.load(OUT_DIR / "chronos_predictions.npz")
    f["chronos"], f["chronos_q80"], f["chronos_q90"] = ch["chronos"], ch["chronos_q80"], ch["chronos_q90"]
    # Seasonal naive without the ACF gate.
    f["seasonal_day_always"] = gather(series, starts - 1440, HORIZON).astype(float)
    f["previous_hour"] = gather(series, starts - HORIZON, HORIZON).astype(float)
    # Trailing quantiles and maxima over 10 and 60 minutes.
    recent = gather(series, starts - 10, 10).astype(float)
    for q in (50, 80, 90, 95, 99):
        f[f"trailing_p{q}"] = np.repeat(np.quantile(recent, q / 100, axis=-1)[:, :, None], HORIZON, axis=2)
    f["trailing_max10"] = np.repeat(recent.max(axis=-1)[:, :, None], HORIZON, axis=2)
    hour = gather(series, starts - 60, 60).astype(float)
    f["trailing_max60"] = np.repeat(hour.max(axis=-1)[:, :, None], HORIZON, axis=2)
    # Intermittent-demand forecasters on the previous day.
    ctx = gather(series, starts - 1440, 1440).astype(float).reshape(-1, 1440)
    f["tsb"] = np.repeat(tsb(ctx).reshape(len(series), len(starts), 1), HORIZON, axis=2)
    f["croston"] = np.repeat(croston(ctx).reshape(len(series), len(starts), 1), HORIZON, axis=2)
    return f


def frame(model, window, variant, meta, sc, memory):
    df = pd.DataFrame(sc)
    df.insert(0, "function_id", meta["function_id"].to_numpy())
    df.insert(1, "class", meta["class"].to_numpy())
    df.insert(2, "model", model)
    df.insert(3, "window", window)
    df.insert(4, "variant", variant)
    df["MBmin"] = df["O"] * memory
    return df


def main() -> None:
    series, meta = load_series()
    starts = origins()
    y = gather(series, starts, HORIZON).astype(float)
    slices = origin_slices(starts)
    attrs = pd.read_csv(ROOT / "data" / "function_attributes.csv").set_index("function_id").loc[meta["function_id"]]
    memory = attrs["memory_mb"].to_numpy()
    duration = attrs["avg_duration_ms"].fillna(attrs["avg_duration_ms"].median()).clip(lower=10.0)
    cap_dur = np.maximum(60000.0 / duration.to_numpy(), 1.0)[:, None, None]
    print("capacity from duration: median", float(np.median(cap_dur)), "min", float(cap_dur.min()), "max", float(cap_dur.max()))
    prev_min = series[:, starts - 1].astype(float)[:, :, None]
    r_prev_full = np.concatenate([np.ceil(prev_min / CAP), np.ceil(y[:, :, :-1] / CAP)], axis=2)
    r_prev_dur = np.concatenate([np.ceil(prev_min / cap_dur), np.ceil(y[:, :, :-1] / cap_dur)], axis=2)

    f = build_forecasts(series, meta, starts)
    # The first LSTM had a ReLU output head with dead units. Keep it as lstm_relu,
    # and use the linear-head LSTM (same loss as before) as the main LSTM.
    f["lstm_relu"] = f["lstm"]
    lin = OUT_DIR / "lstm_lin_tau0.5_seed42.npz"
    if lin.exists():
        f["lstm"] = np.load(lin)["yhat"]
    q90 = OUT_DIR / "lstm_lin_tau0.9_seed42.npz"
    if q90.exists():
        f["lstm_tau0.9"] = np.load(q90)["yhat"]
    # Derived headroom models.
    for m in (1.25, 1.5, 2.0, 3.0):
        f[f"seasonal_x{m}"] = f["seasonal_naive"] * m
    f["seasonal_or_p95"] = np.maximum(f["seasonal_naive"], f["trailing_p95"])
    for base_name in ("ets", "lstm", "chronos"):
        for m in (1.25, 1.5, 2.0):
            f[f"{base_name}_x{m}"] = f[base_name] * m

    # M1 diagnostic: tiny positive forecasts on idle minutes of sparse functions.
    diag = []
    sparse = (meta["class"] == "sparse").to_numpy()
    for name in ("seasonal_naive", "last_value", "trailing_p95", "ets", "lstm", "chronos", "tsb", "croston"):
        yh = f[name][sparse]
        yy = y[sparse]
        tiny = (yh > 0) & (yh < THRESH)
        diag.append({"model": name, "share_minutes_tiny_positive": tiny.mean(), "share_minutes_positive": (yh > 0).mean(),
                     "share_idle_minutes_tiny_positive": tiny[yy == 0].mean()})
    pd.DataFrame(diag).to_csv(OUT_DIR / "diag_tiny_forecasts.csv", index=False)
    print(pd.DataFrame(diag).round(3).to_string(index=False))

    frames = []
    for name, yh in f.items():
        yhf = fix(yh)
        for window, sl in slices.items():
            sc = score(y[:, sl], r_prev_full[:, sl], yhat=yhf[:, sl], memory=memory)
            frames.append(frame(name, window, "C10_thresholded", meta, sc, memory))
            if name in ("seasonal_naive", "last_value", "trailing_p95", "ets", "lstm", "chronos", "chronos_q90"):
                sc = score(y[:, sl], r_prev_full[:, sl], yhat=yh[:, sl], memory=memory)
                frames.append(frame(name, window, "C10_raw", meta, sc, memory))
                sc = score(y[:, sl], r_prev_dur[:, sl], yhat=yhf[:, sl], cap=cap_dur, memory=memory)
                frames.append(frame(name, window, "duration_capacity", meta, sc, memory))
    # Reactive 10-minute keep-alive: warm set is the largest containers used in the previous 10 minutes.
    rc = np.ceil(series.astype(float) / CAP)
    warm_all = np.stack([np.max(np.stack([np.roll(rc, k, axis=1) for k in range(1, 11)]), axis=0)])[0]
    warm = gather(warm_all, starts, HORIZON)
    for window, sl in slices.items():
        sc = score(y[:, sl], r_prev_full[:, sl], p=warm[:, sl], memory=memory)
        frames.append(frame("keepalive10_reactive", window, "C10_thresholded", meta, sc, memory))
    pf = pd.concat(frames, ignore_index=True)
    pf.to_csv(OUT_DIR / "robust_per_function.csv", index=False)

    # Summaries.
    def summary(df):
        both = pd.concat([df, df.assign(**{"class": "all"})], ignore_index=True)
        g = both.groupby(["variant", "window", "model", "class"], sort=False)
        return g.agg(
            smape=("smape", "median"), mae=("mae", "median"), S=("S", "median"), O=("O", "median"), cold=("cold", "median"),
            MBmin=("MBmin", "median"), S_p90=("S", lambda s: s.quantile(0.9)), O_p90=("O", lambda s: s.quantile(0.9)),
            share_short=("S", lambda s: (s > 0).mean()), S_sum=("S", "sum"), O_sum=("O", "sum"), req_sum=("req", "sum"),
        ).reset_index()

    sm = summary(pf)
    sm["S_frac"] = sm["S_sum"] / sm["req_sum"]
    sm["O_frac"] = sm["O_sum"] / sm["req_sum"]
    sm.to_csv(OUT_DIR / "robust_summary.csv", index=False)

    # Class-reweighted fleet medians.
    w_class = {c: CLASS_N[c] / 20 for c in CLASSES}
    rows = []
    for (variant, window, model), g in pf.groupby(["variant", "window", "model"], sort=False):
        w = g["class"].map(w_class).to_numpy()
        rows.append({"variant": variant, "window": window, "model": model,
                     "S_fleet": wmedian(g["S"].to_numpy(), w), "O_fleet": wmedian(g["O"].to_numpy(), w),
                     "cold_fleet": wmedian(g["cold"].to_numpy(), w),
                     "share_short_fleet": float(np.average(g["S"].to_numpy() > 0, weights=w))})
    fleet = pd.DataFrame(rows)
    fleet.to_csv(OUT_DIR / "robust_fleet.csv", index=False)

    # Stratified bootstrap over functions, All median, days 11-14 thresholded C10.
    rng = np.random.default_rng(42)
    sub = pf[(pf.variant == "C10_thresholded") & (pf.window == "days_11_14")]
    models = ["seasonal_naive", "last_value", "trailing_p95", "ets", "lstm", "chronos", "previous_hour", "chronos_q90"]
    idx_by_class = [np.flatnonzero((meta["class"] == c).to_numpy()) for c in CLASSES]
    wide = {m: sub[sub.model == m].set_index("function_id").loc[meta["function_id"]] for m in models}
    draws = np.concatenate([rng.choice(ix, size=(B, len(ix)), replace=True) for ix in idx_by_class], axis=1)
    ci_rows = []
    for m in models:
        for stat in ("smape", "S", "O", "cold", "MBmin"):
            v = wide[m][stat].to_numpy()
            med = np.median(v[draws], axis=1)
            ci_rows.append({"model": m, "stat": stat, "median": float(np.median(v)), "lo": float(np.percentile(med, 2.5)), "hi": float(np.percentile(med, 97.5))})
    for a, b_ in (("trailing_p95", "seasonal_naive"), ("chronos", "seasonal_naive"), ("lstm", "seasonal_naive"), ("ets", "seasonal_naive")):
        for stat in ("smape", "S", "O"):
            d = wide[a][stat].to_numpy() - wide[b_][stat].to_numpy()
            med = np.median(d[draws], axis=1)
            ci_rows.append({"model": f"{a} minus {b_}", "stat": stat, "median": float(np.median(d)), "lo": float(np.percentile(med, 2.5)), "hi": float(np.percentile(med, 97.5))})
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(OUT_DIR / "robust_ci.csv", index=False)

    # Selection rules (days 11-12 pick, days 13-14 check), thresholded C10.
    pool = ["seasonal_naive", "last_value", "trailing_p95", "ets", "lstm", "chronos"]
    sel_rows = []
    s1112 = sm[(sm.variant == "C10_thresholded") & (sm.window == "days_11_12") & sm.model.isin(pool)]
    pick = {}
    for crit in ("shortfall_first", "cost_aware_S_plus_O"):
        for c in CLASSES:
            t = s1112[s1112["class"] == c].copy()
            t["SO"] = t["S"] + t["O"]
            t = t.sort_values(["S", "O", "smape"] if crit == "shortfall_first" else ["SO", "S", "smape"])
            pick[(crit, c)] = t.iloc[0]["model"]
        sel_rows.append({"criterion": crit, **{c: pick[(crit, c)] for c in CLASSES}})
    print(pd.DataFrame(sel_rows).to_string(index=False))
    held = pf[(pf.variant == "C10_thresholded") & (pf.window == "days_13_14")]
    out = []
    for crit in ("shortfall_first", "cost_aware_S_plus_O"):
        parts = [held[(held["class"] == c) & (held.model == pick[(crit, c)])] for c in CLASSES]
        chosen = pd.concat(parts)
        out.append({"policy": crit, "smape": chosen.smape.median(), "S": chosen.S.median(), "O": chosen.O.median()})
    for m in pool:
        g = held[held.model == m]
        out.append({"policy": f"always_{m}", "smape": g.smape.median(), "S": g.S.median(), "O": g.O.median()})
    # Paired bootstrap of the two class picks against single models on days 13-14.
    held_wide = {m: held[held.model == m].set_index("function_id").loc[meta["function_id"]] for m in pool}
    cls = meta["class"].to_numpy()
    pol = {}
    for crit in ("shortfall_first", "cost_aware_S_plus_O"):
        cols = {}
        for stat in ("smape", "S", "O", "cold"):
            cols[stat] = np.array([held_wide[pick[(crit, cls[i])]][stat].to_numpy()[i] for i in range(len(cls))])
        pol[crit] = cols
    for m in pool:
        pol[f"always_{m}"] = {stat: held_wide[m][stat].to_numpy() for stat in ("smape", "S", "O", "cold")}
    pol_rows = []
    for name, cols in pol.items():
        row = {"policy": name}
        for stat in ("smape", "S", "O", "cold"):
            med = np.median(cols[stat][draws], axis=1)
            row[stat] = float(np.median(cols[stat]))
            row[stat + "_lo"], row[stat + "_hi"] = float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))
        pol_rows.append(row)
    for crit in ("shortfall_first", "cost_aware_S_plus_O"):
        for ref in ("always_seasonal_naive", "always_trailing_p95"):
            row = {"policy": f"{crit} minus {ref}"}
            for stat in ("smape", "S", "O", "cold"):
                d = pol[crit][stat] - pol[ref][stat]
                med = np.median(d[draws], axis=1)
                row[stat] = float(np.median(d))
                row[stat + "_lo"], row[stat + "_hi"] = float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))
            pol_rows.append(row)
    pd.DataFrame(pol_rows).to_csv(OUT_DIR / "robust_policy_ci.csv", index=False)
    print(pd.DataFrame(pol_rows).round(1).to_string(index=False))
    pd.DataFrame(sel_rows).to_csv(OUT_DIR / "robust_selection_pick.csv", index=False)
    pd.DataFrame(out).to_csv(OUT_DIR / "robust_selection_heldout.csv", index=False)
    print(pd.DataFrame(out).round(2).to_string(index=False))

    (OUT_DIR / "robust_meta.json").write_text(json.dumps({"threshold": THRESH, "capacity": CAP, "bootstrap": B}), encoding="utf-8")


if __name__ == "__main__":
    main()
