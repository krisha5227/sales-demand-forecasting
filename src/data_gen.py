"""Synthetic weekly sales generator (clearly synthetic - fixed seed).

Demand for product i in week t is built multiplicatively:

    mu = base_i * trend_i(t) * seasonality_i(t) * holiday_i(t)
              * promo_lift_i(t) * (price_t / price_0)^(-elasticity_i)

    units ~ Poisson( mu * LogNormal(0, 0.12) )          # over-dispersed count noise

Realistic data problems are then injected on purpose so the cleaning step has real work:
missing weeks, negative values (returns booked as sales) and outlier spikes.
The un-corrupted demand and the injected-issue labels are saved separately
(``synthetic_ground_truth.csv``) and are used only for a data-quality sanity report,
never as a model input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .features import CATEGORIES, holiday_flag_for_weeks

# category -> (peak week of year, seasonal amplitude, holiday uplift)
CAT_PARAMS = {
    "Beverages":    (28, 0.30, 0.10),   # summer peak
    "Snacks":       (48, 0.15, 0.20),
    "Household":    (12, 0.10, 0.05),   # spring-cleaning bump
    "PersonalCare": (2,  0.12, 0.08),
    "Seasonal":     (51, 0.55, 0.35),   # strong winter-holiday peak
}


def generate(seed: int = C.SEED, n_products: int = C.N_PRODUCTS, n_weeks: int = C.N_WEEKS):
    rng = np.random.default_rng(seed)
    weeks = pd.date_range(C.START_DATE, periods=n_weeks, freq="W-MON")
    t = np.arange(n_weeks)
    woy = weeks.isocalendar().week.to_numpy().astype(float)
    hol = holiday_flag_for_weeks(weeks).astype(float)

    rows, truth = [], []
    for i in range(n_products):
        pid = f"P{i + 1:03d}"
        cat = CATEGORIES[i % len(CATEGORIES)]
        peak, amp, hol_up = CAT_PARAMS[cat]

        # every 8th product is a slow mover (~1-3 units/week) so zero-demand weeks genuinely occur
        if i % 8 == 7:
            base = rng.uniform(1, 3)
        else:
            base = float(np.clip(rng.lognormal(mean=4.0, sigma=0.8), 12, 500))

        trend = np.exp(rng.normal(0.05, 0.08) * t / 52.18)                      # yearly growth
        phase = peak + rng.normal(0, 3)
        a = amp * rng.uniform(0.7, 1.3)
        ang = 2 * np.pi * (woy - phase) / 52.18
        season = 1 + a * np.cos(ang) + 0.3 * a * np.cos(2 * ang)
        holiday = 1 + hol * (hol_up * rng.uniform(0.5, 1.5))

        # price path: list price steps up ~3% a couple of times a year, promos cut it
        price0 = float(np.clip(rng.lognormal(2.0, 0.5), 2, 40))
        list_price = price0 * np.cumprod(1 + 0.03 * (rng.random(n_weeks) < 0.04))
        promo = np.zeros(n_weeks)
        on = 0
        for k in range(n_weeks):                                                # 1-2 week promos
            on = 1 if (on and rng.random() < 0.35) else int(rng.random() < 0.06)
            promo[k] = on
        depth = np.where(promo == 1, rng.uniform(0.10, 0.30, size=n_weeks), 0.0)
        price = np.round(list_price * (1 - depth), 2)
        elasticity = rng.uniform(0.8, 1.8)
        promo_lift = np.where(promo == 1, rng.uniform(1.10, 1.35), 1.0)
        price_effect = (price / price0) ** (-elasticity)

        mu = base * trend * season * holiday * promo_lift * price_effect
        units = rng.poisson(mu * rng.lognormal(0, 0.12, size=n_weeks)).astype(float)

        issue = np.array([""] * n_weeks, dtype=object)
        obs = units.copy()
        # negatives: returns booked as negative sales (0.15%)
        neg = rng.random(n_weeks) < 0.0015
        obs[neg] = -rng.integers(1, 10, size=neg.sum())
        issue[neg] = "negative"
        # outliers: x4-x10 spikes (0.3%) - keycode / bulk order
        out = (rng.random(n_weeks) < 0.003) & ~neg
        obs[out] = np.round(units[out] * rng.uniform(4, 10, size=out.sum()) + 5)
        issue[out] = "outlier"
        # missing: value blank (0.5%) - the other missing weeks are whole absent rows (below)
        blank = (rng.random(n_weeks) < 0.005) & ~neg & ~out
        obs[blank] = np.nan
        issue[blank] = "missing_value"
        absent = (rng.random(n_weeks) < 0.005) & (t > 3) & ~neg & ~out & ~blank
        issue[absent] = "missing_row"

        df = pd.DataFrame({"product_id": pid, "category": cat, "week_start": weeks,
                           "units_sold": obs, "price": price, "promo_flag": promo.astype(int)})
        tr = pd.DataFrame({"product_id": pid, "week_start": weeks, "true_units": units,
                           "injected_issue": issue})
        rows.append(df.loc[~absent])
        truth.append(tr)

    raw = pd.concat(rows, ignore_index=True)
    truth = pd.concat(truth, ignore_index=True)
    return raw, truth


def main():
    C.DATA_DIR.mkdir(exist_ok=True, parents=True)
    raw, truth = generate()
    raw.to_csv(C.RAW_FILE, index=False)
    truth.to_csv(C.GROUND_TRUTH_FILE, index=False)
    print(f"[data_gen] SYNTHETIC data written: {len(raw)} rows, "
          f"{raw.product_id.nunique()} products, {raw.week_start.nunique()} weeks")
    print(truth["injected_issue"].replace("", "none").value_counts().to_string())


if __name__ == "__main__":
    main()
