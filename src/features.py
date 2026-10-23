"""Calendar helpers and leakage-safe feature engineering.

Forecast set-up (direct multi-horizon strategy)
-----------------------------------------------
At forecast *origin* ``o`` (the end of week ``o``) we know demand up to and including
week ``o``. For each horizon ``h`` in {1, 2, 3} we predict the demand of target week
``t = o + h``. One model is trained per horizon, so no predicted values are ever fed back
in as inputs.

What is allowed as an input at origin ``o`` for target ``t``
  * demand history up to ``o`` (lags, rolling stats, trailing level)
  * the 52-week-ago demand of the target week, y[t-52] (always <= o because h <= 3)
  * calendar facts about ``t`` (week of year, month, holiday flag): known in advance
  * planned promotion flag and planned price of week ``t``: assumed known in advance
    (promo calendars and price lists are set before the week starts) - an assumption
    that is documented in the README.

Lag naming: ``lag_k`` is the k-th most recent observation *at the origin*
(lag_1 = y[o], lag_2 = y[o-1], lag_4 = y[o-3]). For h = 1 these are exactly the usual
lag-1/2/4 of the target week. ``lag_52`` is y[t-52].

Scale normalisation: products differ a lot in volume, so all demand-derived features and
the target are divided by the trailing 13-week mean ("level") at the origin. Predictions
are multiplied back by the same level. The level uses only data up to the origin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dateutil.easter import easter

CATEGORIES = ["Beverages", "Snacks", "Household", "PersonalCare", "Seasonal"]


# --------------------------------------------------------------------------- calendar
def holiday_week_starts(years) -> set:
    """Mondays of the weeks that contain a (US-style retail) holiday.

    New Year, Easter, Memorial Day, July 4th, Labor Day, Thanksgiving, Christmas.
    """
    days = []
    for y in years:
        days.append(pd.Timestamp(y, 1, 1))
        days.append(pd.Timestamp(easter(y)))
        may = pd.date_range(f"{y}-05-01", f"{y}-05-31")
        days.append(may[may.dayofweek == 0][-1])                       # last Monday of May
        days.append(pd.Timestamp(y, 7, 4))
        sep = pd.date_range(f"{y}-09-01", f"{y}-09-30")
        days.append(sep[sep.dayofweek == 0][0])                        # first Monday of Sep
        nov = pd.date_range(f"{y}-11-01", f"{y}-11-30")
        days.append(nov[nov.dayofweek == 3][3])                        # 4th Thursday of Nov
        days.append(pd.Timestamp(y, 12, 25))
    return {(d - pd.Timedelta(days=int(d.dayofweek))).normalize() for d in days}


def holiday_flag_for_weeks(week_starts: pd.DatetimeIndex) -> np.ndarray:
    years = range(week_starts.year.min() - 1, week_starts.year.max() + 2)
    hs = holiday_week_starts(years)
    return np.array([1 if w.normalize() in hs else 0 for w in week_starts], dtype=int)


# --------------------------------------------------------------------------- panel arrays
class Panel:
    """Wide (product x week) numpy view of the cleaned data."""

    def __init__(self, clean: pd.DataFrame):
        clean = clean.sort_values(["product_id", "week_start"])
        self.products = list(clean["product_id"].drop_duplicates())
        self.weeks = pd.DatetimeIndex(sorted(clean["week_start"].unique()))
        P, T = len(self.products), len(self.weeks)

        def wide(col):
            return clean.pivot(index="product_id", columns="week_start", values=col) \
                        .loc[self.products, self.weeks].to_numpy(dtype=float)

        self.y = wide("units")
        self.price = wide("price")
        self.promo = wide("promo_flag")
        self.imputed = wide("is_imputed")
        self.hol = holiday_flag_for_weeks(self.weeks).astype(float)
        cat = clean.drop_duplicates("product_id").set_index("product_id").loc[self.products, "category"]
        self.category = cat.to_numpy()
        self.P, self.T = P, T

    def copy_with(self, y=None, price=None, promo=None) -> "Panel":
        new = object.__new__(Panel)
        new.__dict__.update(self.__dict__)
        if y is not None:
            new.y = y
        if price is not None:
            new.price = price
        if promo is not None:
            new.promo = promo
        return new


# --------------------------------------------------------------------------- features
def feature_columns() -> list[str]:
    base = ["lag_1", "lag_2", "lag_4", "lag_52", "lag52_smooth",
            "roll_mean_4", "roll_std_4", "roll_mean_8", "roll_std_8",
            "log_level", "week_of_year", "month", "woy_sin", "woy_cos",
            "holiday_flag", "promo_flag", "promo_last", "price_ratio"]
    return base + [f"cat_{c}" for c in CATEGORIES]


def build_features(panel: Panel, h: int, t_min: int | None = None) -> pd.DataFrame:
    """Build one row per (product, target week) for horizon ``h``.

    Every demand-derived value is computed from ``panel.y[:, :o+1]`` with ``o = t - h``
    (or from y[t-52], which is <= o). ``y_true`` is stored only as the label.
    """
    from .config import WARMUP
    t_min = WARMUP if t_min is None else t_min
    P, T = panel.P, panel.T
    ts = np.arange(t_min, T)
    os_ = ts - h
    assert os_.min() >= 12 and (ts - 53).min() >= 0

    def at(mat, idx):                         # mat[:, idx] -> (P, len(idx))
        return mat[:, idx]

    y = panel.y
    # trailing windows ending at the origin (inclusive)
    def window(mat, length):
        return np.stack([mat[:, o - length + 1:o + 1] for o in os_], axis=1)   # (P, n, length)

    w4, w8, w13 = window(y, 4), window(y, 8), window(y, 13)
    level = np.maximum(w13.mean(axis=2), 0.5)                                  # (P, n)

    feats = {
        "lag_1": at(y, os_) / level,
        "lag_2": at(y, os_ - 1) / level,
        "lag_4": at(y, os_ - 3) / level,
        "lag_52": at(y, ts - 52) / level,
        "lag52_smooth": (at(y, ts - 53) + at(y, ts - 52) + at(y, ts - 51)) / 3.0 / level,
        "roll_mean_4": w4.mean(axis=2) / level,
        "roll_std_4": w4.std(axis=2, ddof=1) / level,
        "roll_mean_8": w8.mean(axis=2) / level,
        "roll_std_8": w8.std(axis=2, ddof=1) / level,
        "log_level": np.log(level),
        "holiday_flag": np.tile(panel.hol[ts], (P, 1)),
        "promo_flag": at(panel.promo, ts),
        "promo_last": at(panel.promo, os_),
    }
    # price of the target week relative to the recent (13-wk) median list price at the origin
    price_ref = np.median(window(panel.price, 13), axis=2)
    feats["price_ratio"] = at(panel.price, ts) / price_ref

    wk = panel.weeks[ts]
    woy = wk.isocalendar().week.to_numpy().astype(float)
    feats["week_of_year"] = np.tile(woy, (P, 1))
    feats["month"] = np.tile(wk.month.to_numpy().astype(float), (P, 1))
    feats["woy_sin"] = np.tile(np.sin(2 * np.pi * woy / 52.18), (P, 1))
    feats["woy_cos"] = np.tile(np.cos(2 * np.pi * woy / 52.18), (P, 1))
    for c in CATEGORIES:
        feats[f"cat_{c}"] = np.tile((panel.category == c).astype(float)[:, None], (1, len(ts)))

    n = len(ts)
    out = pd.DataFrame({k: v.reshape(-1) for k, v in feats.items()})
    out.insert(0, "product_id", np.repeat(panel.products, n))
    out.insert(1, "target_idx", np.tile(ts, P))
    out.insert(2, "origin_idx", np.tile(os_, P))
    out.insert(3, "level", level.reshape(-1))
    out.insert(4, "y_true", at(y, ts).reshape(-1))
    out.insert(5, "y_norm", (at(y, ts) / level).reshape(-1))
    out.insert(6, "is_imputed_target", at(panel.imputed, ts).reshape(-1))
    out["horizon"] = h
    return out


# --------------------------------------------------------------------------- splits
def make_splits(T: int, val_weeks: int, test_weeks: int, warmup: int) -> dict:
    """Strict chronological split on target-week indices (no shuffling anywhere)."""
    t0 = T - test_weeks
    v0 = t0 - val_weeks
    return {"train": (warmup, v0 - 1), "val": (v0, t0 - 1), "test": (t0, T - 1)}


def rows_for_training(df: pd.DataFrame, first_eval_target: int, h: int) -> pd.DataFrame:
    """Training rows whose *label is known* when the first forecast is issued.

    The first forecast of an evaluation block (first target week F) is made at origin
    F - h, when demand through week F - h is known. So training rows must have target
    <= F - h (their labels exist by then). This purges the h-1 most recent weeks.
    """
    sub = df[df["target_idx"] <= first_eval_target - h]
    assert sub["target_idx"].max() <= first_eval_target - h
    return sub


# --------------------------------------------------------------------------- leakage checks
def assert_no_leakage(panel: Panel, h: int, seed: int = 0, n_cutoffs: int = 5) -> dict:
    """Perturbation test for look-ahead leakage.

    For several cut-off weeks ``c``: replace all demand after ``c`` with garbage (and
    scramble planned price / promo after c + h, the furthest target any checked row can
    have). Re-build the features. Every row whose forecast origin is <= c must have
    *identical* feature values. If any feature peeked at demand after its origin, it
    would change and the assertion would fail.
    """
    rng = np.random.default_rng(seed)
    base = build_features(panel, h)
    fcols = feature_columns()
    T = panel.T
    cutoffs = rng.integers(70, T - 10, size=n_cutoffs)
    n_checked = 0
    for c in cutoffs:
        y2 = panel.y.copy()
        y2[:, c + 1:] = rng.uniform(1e4, 1e5, size=y2[:, c + 1:].shape)
        pr2 = panel.price.copy()
        pr2[:, c + h + 1:] = rng.uniform(1, 1e3, size=pr2[:, c + h + 1:].shape)
        pm2 = panel.promo.copy()
        pm2[:, c + h + 1:] = 1.0 - pm2[:, c + h + 1:]
        pert = build_features(panel.copy_with(y=y2, price=pr2, promo=pm2), h)
        mask = (base["origin_idx"] <= c).to_numpy()
        a = base.loc[mask, fcols].to_numpy()
        b = pert.loc[mask, fcols].to_numpy()
        if not np.allclose(a, b, equal_nan=True):
            bad = [col for col in fcols if not np.allclose(base.loc[mask, col], pert.loc[mask, col])]
            raise AssertionError(f"LEAKAGE: features changed when future data was altered: {bad}")
        n_checked += int(mask.sum())
    # the label must never be a feature
    assert not ({"y_true", "y_norm"} & set(fcols))
    return {"horizon": h, "cutoffs_tested": int(len(cutoffs)), "rows_checked": n_checked, "passed": True}
