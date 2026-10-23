"""Cleaning raw weekly sales -> a complete, causal, flagged product x week panel.

Decisions (also documented in the README):
1. Full grid      : re-index to every product x every week. Absent rows become NaN.
2. Duplicates     : none expected; if (product, week) repeats, units are summed.
3. Negative units : treated as returns / booking errors, NOT as demand -> set to NaN
                    (clipping to 0 would fabricate "zero demand" weeks).
4. Outliers       : flagged with a CAUSAL robust rule (only the previous 13 weeks are used,
                    so cleaning never looks into the future). A value is an outlier when
                    it exceeds  median + 5 * robust_sigma  and, in promo/holiday weeks
                    (where genuine spikes are expected), also exceeds 3.5 x median.
                    Outliers are winsorised to that upper bound, not deleted.
5. Missing values : causal forward-fill (last cleaned value, max 4 weeks); the rare series
                    start gap is back-filled. Every filled week is flagged ``is_imputed``
                    and is EXCLUDED from all accuracy / stock-out metrics, so the models
                    are never scored against made-up numbers.
6. Zero demand   : kept as genuine zeros (slow movers). MAPE excludes them; WAPE keeps them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .features import holiday_flag_for_weeks


def _causal_outlier_bound(x: np.ndarray, event: np.ndarray, window: int = 13, min_obs: int = 6):
    """Upper bound per week from the preceding ``window`` weeks (never uses week t itself)."""
    n = len(x)
    bound = np.full(n, np.inf)
    for t in range(n):
        past = x[max(0, t - window):t]
        past = past[~np.isnan(past)]
        if len(past) < min_obs:
            continue
        med = np.median(past)
        mad = np.median(np.abs(past - med))
        sigma = max(1.4826 * mad, np.sqrt(med), 1.0)
        ub = med + 5 * sigma
        if event[t]:
            ub = max(ub, 3.5 * med)
        bound[t] = ub
    return bound


def clean(raw: pd.DataFrame, verbose: bool = True):
    raw = raw.copy()
    raw["week_start"] = pd.to_datetime(raw["week_start"])
    n_dupe = int(raw.duplicated(["product_id", "week_start"]).sum())
    if n_dupe:   # sum duplicate bookings (min_count=1 keeps all-NaN groups as NaN)
        raw = raw.groupby(["product_id", "category", "week_start"], as_index=False).agg(
            units_sold=("units_sold", lambda s: s.sum(min_count=1)),
            price=("price", "mean"), promo_flag=("promo_flag", "max"))

    weeks = pd.date_range(raw["week_start"].min(), raw["week_start"].max(), freq="W-MON")
    prods = sorted(raw["product_id"].unique())
    grid = pd.MultiIndex.from_product([prods, weeks], names=["product_id", "week_start"])
    df = raw.set_index(["product_id", "week_start"]).reindex(grid).reset_index()
    cat = raw.drop_duplicates("product_id").set_index("product_id")["category"]
    df["category"] = df["product_id"].map(cat)
    n_absent_rows = int(df["units_sold"].isna().sum() - raw["units_sold"].isna().sum())

    # planned price / promo for rows that were absent: carry price forward, assume no promo
    df["price"] = df.groupby("product_id")["price"].transform(lambda s: s.ffill().bfill())
    df["promo_flag"] = df["promo_flag"].fillna(0).astype(int)

    # 3. negatives -> NaN
    neg = df["units_sold"] < 0
    df.loc[neg, "units_sold"] = np.nan
    df["was_negative"] = neg.astype(int)

    # 4. causal outliers (winsorise)
    hol = pd.Series(holiday_flag_for_weeks(weeks), index=weeks)
    df["is_outlier"] = 0
    pieces = []
    for pid, g in df.groupby("product_id", sort=False):
        g = g.sort_values("week_start").copy()
        x = g["units_sold"].to_numpy(dtype=float)
        event = (g["promo_flag"].to_numpy() == 1) | (hol.loc[g["week_start"]].to_numpy() == 1)
        bound = _causal_outlier_bound(x, event)
        flag = x > bound
        x2 = np.where(flag, bound, x)
        g["units"] = np.round(x2)
        g["is_outlier"] = flag.astype(int)
        pieces.append(g)
    df = pd.concat(pieces, ignore_index=True)

    # 5. causal forward fill of remaining gaps, flagged
    df["is_imputed"] = df["units"].isna().astype(int)
    df["units"] = df.groupby("product_id")["units"].transform(lambda s: s.ffill(limit=4).bfill())
    df["units"] = df["units"].fillna(0)
    assert df["units"].notna().all() and (df["units"] >= 0).all()

    report = {
        "rows_after_reindex": int(len(df)),
        "duplicate_rows_summed": n_dupe,
        "absent_rows_added": n_absent_rows,
        "blank_values": int(raw["units_sold"].isna().sum()),
        "negative_values": int(neg.sum()),
        "outliers_winsorised": int(df["is_outlier"].sum()),
        "imputed_weeks_flagged": int(df["is_imputed"].sum()),
        "zero_demand_share_pct": round(100 * float((df["units"] == 0).mean()), 2),
    }
    if verbose:
        print("[data_prep]", report)
    cols = ["product_id", "category", "week_start", "units", "price", "promo_flag",
            "is_imputed", "is_outlier", "was_negative"]
    return df[cols].sort_values(["product_id", "week_start"]).reset_index(drop=True), report


def quality_vs_truth(clean_df: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Sanity report (synthetic data only): did cleaning catch what was injected?"""
    t = truth.copy()
    t["week_start"] = pd.to_datetime(t["week_start"])
    m = clean_df.merge(t, on=["product_id", "week_start"], how="left")
    out = {}
    inj = m["injected_issue"] == "outlier"
    out["injected_outliers"] = int(inj.sum())
    out["injected_outliers_flagged"] = int((inj & (m["is_outlier"] == 1)).sum())
    out["false_outlier_flags"] = int(((~inj) & (m["is_outlier"] == 1)).sum())
    ok = m["is_imputed"] == 0
    out["MAE_clean_vs_true_demand_all_weeks"] = round(float((m["units"] - m["true_units"]).abs().mean()), 3)
    out["MAE_clean_vs_true_demand_non_imputed"] = round(float((m.loc[ok, "units"] - m.loc[ok, "true_units"]).abs().mean()), 3)
    return out
