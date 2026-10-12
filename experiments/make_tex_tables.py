"""Write LaTeX table bodies from results/robust_*.csv so no number is typed by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import OUT_DIR, ROOT

TABLES = ROOT / "tables"
TABLES.mkdir(exist_ok=True)
CLASSES = ["sparse", "daily", "bursty", "other", "all"]
V = "C10_thresholded"


def fmt(x, digits=1):
    if pd.isna(x):
        return "--"
    if abs(x) >= 1000:
        return f"{x:,.0f}".replace(",", "{,}")
    return f"{x:.{digits}f}"


def main() -> None:
    sm = pd.read_csv(OUT_DIR / "robust_summary.csv")
    pf = pd.read_csv(OUT_DIR / "robust_per_function.csv")
    ci = pd.read_csv(OUT_DIR / "robust_ci.csv")
    pick = pd.read_csv(OUT_DIR / "robust_selection_pick.csv").set_index("criterion")

    # Table I: by class.
    names = [("seasonal_naive", "Seasonal naive"), ("previous_hour", "Previous hour"), ("last_value", "Last value"),
             ("trailing_p95", "Trailing P95"), ("ets", "ETS"), ("lstm", "LSTM"), ("chronos", "Chronos")]
    t = sm[(sm.variant == V) & (sm.window == "days_11_14")]
    lines = []
    for key, label in names:
        for i, (stat, name) in enumerate((("smape", "sMAPE (\\%)"), ("S", "Short"), ("O", "Extra"))):
            cells = [fmt(t[(t.model == key) & (t["class"] == c)][stat].iloc[0]) for c in CLASSES]
            lines.append(f"{label if i == 0 else ''} & {name} & " + " & ".join(cells) + " \\\\")
        lines.append("\\hline")
    (TABLES / "tab_byclass.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Table II: all functions, extended columns.
    names2 = [("seasonal_naive", "Seasonal naive"), ("previous_hour", "Previous hour"), ("last_value", "Last value"),
              ("trailing_p95", "Trailing P95"), ("ets", "ETS"), ("lstm", "LSTM"), ("lstm_tau0.9", "LSTM, tau 0.9"),
              ("chronos", "Chronos"), ("chronos_q90", "Chronos q90"), ("tsb", "TSB"), ("croston", "Croston"),
              ("keepalive10_reactive", "Keep-alive 10 min$^{*}$")]
    a = t[t["class"] == "all"].set_index("model")
    lines = []
    for key, label in names2:
        if key not in a.index:
            continue
        r = a.loc[key]
        lines.append(" & ".join([label, fmt(r.smape), fmt(r.S), fmt(r.O), fmt(r.cold), fmt(r.MBmin / 1000, 1), f"{100 * r.share_short:.0f}"]) + " \\\\")
    lines.append("\\hline")
    (TABLES / "tab_all.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Table III: policies on days 11-12 and 13-14.
    base = pf[pf.variant == V]

    def policy_values(window, crit):
        parts = []
        for c in ["sparse", "daily", "bursty", "other"]:
            parts.append(base[(base.window == window) & (base["class"] == c) & (base.model == pick.loc[crit, c])])
        return pd.concat(parts)

    rows = []
    def add(label, getter):
        cells = []
        for window in ("days_11_12", "days_13_14"):
            g = getter(window)
            cells += [fmt(g.smape.median()), fmt(g.S.median()), fmt(g.O.median())]
        rows.append(" & ".join([label] + cells) + " \\\\")

    for key, label in [("seasonal_naive", "Seasonal naive"), ("trailing_p95", "Trailing P95"), ("ets", "ETS"),
                       ("lstm", "LSTM"), ("chronos", "Chronos")]:
        add(label, lambda w, k=key: base[(base.window == w) & (base.model == k)])
    rows.append("\\hline")
    add("Pick: shortfall first", lambda w: policy_values(w, "shortfall_first"))
    add("Pick: short plus extra", lambda w: policy_values(w, "cost_aware_S_plus_O"))
    rows.append("\\hline")
    (TABLES / "tab_policy.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    print((TABLES / "tab_byclass.tex").read_text())
    print((TABLES / "tab_all.tex").read_text())
    print((TABLES / "tab_policy.tex").read_text())
    print(pick)


if __name__ == "__main__":
    main()
