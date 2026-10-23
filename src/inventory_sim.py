"""Reorder-point / order-up-to policy and a week-by-week inventory simulation.

Policy: periodic review (R = 1 week), lead time L weeks, lost sales (no back-orders).
At the end of week o (the forecast origin) for each product:

    S_o = sum_{h=1..L+R} forecast(o+h)  +  z * sigma_LR          (order-up-to level)
    inventory position IP_o = on-hand + on-order
    order quantity  q_o = max(0, ceil(S_o - IP_o))                (arrives L weeks later)

sigma_LR is the std-dev of the (L+R)-week aggregate forecast error of THAT forecaster for
THAT product, measured on the validation block (out-of-sample, before the holdout).
A more accurate forecaster therefore needs less safety stock for the same service level.
z = Phi^-1(service level).

Metrics skip the first L weeks (set only by identical initial conditions).
Timeline of week t: receive arrivals -> demand occurs -> sales = min(stock, demand),
unmet demand is lost -> review: place the order for week t+L+1 arrival.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from . import config as C


def aggregate_errors(preds: pd.DataFrame, col: str, truth: pd.DataFrame, L: int, R: int,
                     block: tuple) -> pd.DataFrame:
    """Per-product (L+R)-week aggregate forecast error for each origin in the block."""
    H = L + R
    d = preds.merge(truth[["product_id", "target_idx", "horizon", "y_true", "is_imputed_target"]],
                    on=["product_id", "target_idx", "horizon"])
    d = d[d["horizon"] <= H]
    g = d.groupby(["product_id", "origin_idx"]).agg(
        n=("horizon", "count"), fc=(col, "sum"), act=("y_true", "sum"),
        imp=("is_imputed_target", "sum"), first_t=("target_idx", "min"), last_t=("target_idx", "max")
    ).reset_index()
    g = g[(g["n"] == H) & (g["imp"] == 0) & (g["first_t"] >= block[0]) & (g["last_t"] <= block[1])]
    g["err"] = g["fc"] - g["act"]
    return g


def sigma_by_product(err_df: pd.DataFrame) -> pd.Series:
    return err_df.groupby("product_id")["err"].std(ddof=1).fillna(0.0)


def to_array(preds: pd.DataFrame, col: str, products: list, t_lo: int, t_hi: int, H: int):
    """F[p, o - o_min, h-1] = forecast issued at origin o for week o+h."""
    o_min = t_lo - 1
    n_o = t_hi - o_min + 1
    F = np.full((len(products), n_o, H), np.nan)
    pidx = {p: i for i, p in enumerate(products)}
    d = preds[(preds["horizon"] <= H)]
    pi = d["product_id"].map(pidx).to_numpy()
    oi = d["origin_idx"].to_numpy() - o_min
    ok = (oi >= 0) & (oi < n_o)
    F[pi[ok], oi[ok], d["horizon"].to_numpy()[ok] - 1] = d[col].to_numpy()[ok]
    return F, o_min


def simulate(demand: np.ndarray, price: np.ndarray, F: np.ndarray, o_min: int, t_lo: int,
             sigma: np.ndarray, init_rate: np.ndarray, L: int, R: int, z: float) -> dict:
    """Simulate holdout weeks t_lo .. t_lo + W - 1.

    demand/price: (P, W) arrays for the holdout weeks. F: forecasts (see to_array).
    init_rate: per-product starting weekly rate (identical for every forecaster).
    """
    P, W = demand.shape
    H = L + R
    on_hand = np.ceil(init_rate * R)                       # identical starting stock: one review period of cover
    pipeline = np.zeros((P, W + L + 2))                    # arrivals by holdout week index
    for k in range(L):                                     # orders placed 3 and 2 weeks before the holdout
        pipeline[:, k] = np.ceil(init_rate)
    lost = np.zeros((P, W))
    end_stock = np.zeros((P, W))
    ordered = np.zeros((P, W + 1))                         # column w+1 = order placed at end of week w

    def review(w, stock):
        """Order placed at the end of week w (w = -1: end of the last pre-holdout week)."""
        if w + L + 1 <= W - 1:                             # later orders arrive after the horizon
            fc = F[:, (t_lo + w) - o_min, :H]
            assert not np.isnan(fc).any(), "missing forecasts for the order decision"
            S = np.ceil(fc.sum(axis=1) + z * sigma)
            ip = stock + pipeline[:, w + 1:].sum(axis=1)
            q = np.maximum(0, S - ip)
            pipeline[:, w + L + 1] += q
            ordered[:, w + 1] = q

    review(-1, on_hand)
    for w in range(W):
        on_hand = on_hand + pipeline[:, w]                 # 1) arrivals
        sales = np.minimum(on_hand, demand[:, w])          # 2) demand
        lost[:, w] = demand[:, w] - sales
        on_hand = on_hand - sales
        end_stock[:, w] = on_hand
        review(w, on_hand)                                 # 3) review / order
    # Weeks 0..L-1 are driven only by the initial conditions (orders placed before the holdout
    # arrive then), identical for every forecaster, so they carry no policy information: dropped.
    return {"lost": lost[:, L:], "end_stock": end_stock[:, L:], "ordered": ordered, "warmup_weeks": L}


def summarise(sim: dict, demand: np.ndarray, price: np.ndarray) -> dict:
    lost, stock = sim["lost"], sim["end_stock"]
    hold = (stock * price * C.HOLDING_RATE_ANNUAL / 52.0).sum()
    stockout_cost = (lost * price * C.LOST_MARGIN).sum()
    return {
        "stockout_freq_%": float((lost > 0).mean() * 100),
        "stockout_events": int((lost > 0).sum()),
        "fill_rate_%": float((1 - lost.sum() / demand.sum()) * 100),
        "units_lost": float(lost.sum()),
        "avg_inventory_units": float(stock.mean()),
        "holding_cost": float(hold),
        "lost_margin_cost": float(stockout_cost),
        "total_cost_proxy": float(hold + stockout_cost),
    }


def run_policy(name_cols: dict, preds: pd.DataFrame, val_preds: pd.DataFrame, truth_val: pd.DataFrame,
               products: list, demand: np.ndarray, price: np.ndarray, t_lo: int, val_block: tuple,
               init_rate: np.ndarray, L=C.LEAD_TIME, R=C.REVIEW_PERIOD, sl=C.SERVICE_LEVEL,
               common_sigma_from: str | None = None):
    """Run every forecaster through the identical policy; return metrics and raw sims."""
    z = NormalDist().inv_cdf(sl)
    W = demand.shape[1]
    out, sims, sigmas = {}, {}, {}
    for key, col in name_cols.items():
        sig_src = common_sigma_from or key
        err = aggregate_errors(val_preds, name_cols[sig_src], truth_val, L, R, val_block)
        sig = sigma_by_product(err).reindex(products).fillna(0.0).to_numpy()
        F, o_min = to_array(preds, col, products, t_lo, t_lo + W - 1, L + R)
        sim = simulate(demand, price, F, o_min, t_lo, sig, init_rate, L, R, z)
        out[key], sims[key], sigmas[key] = summarise(sim, demand[:, L:], price[:, L:]), sim, sig
    return out, sims, sigmas


def bootstrap_reduction(lost_base: np.ndarray, lost_model: np.ndarray, B=C.BOOTSTRAP_B, seed=C.SEED):
    """95% CI of the % reduction in stock-out events, resampling products."""
    rng = np.random.default_rng(seed)
    a, b = (lost_base > 0).sum(axis=1), (lost_model > 0).sum(axis=1)
    P = len(a)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, P, P)
        if a[idx].sum() > 0:
            vals.append((1 - b[idx].sum() / a[idx].sum()) * 100)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def pct_reduction(base: float, new: float) -> float:
    return float((1 - new / base) * 100) if base else float("nan")
