"""Metrics, comparison tables, holdout lock and charts.

Metric definitions
------------------
RMSE : sqrt(mean((y - yhat)^2)), in units.
MAPE : mean(|y - yhat| / y) * 100 over rows with y > 0 ONLY. Percentage error is undefined
       at y = 0, so zero-demand weeks are excluded from MAPE (their count is reported).
WAPE : sum|y - yhat| / sum(y) * 100. Well defined with zeros and not dominated by tiny
       denominators, so it is the metric to trust for relative accuracy.
Rows whose target week was imputed (missing in the raw data) are always excluded.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

# validated categorical slots (blue, orange) + neutral greys / ink
BLUE, ORANGE, GREY, INK, GRID = "#2a78d6", "#eb6834", "#8a8984", "#2b2a28", "#e4e3df"
SURFACE = "#fcfcfb"


# --------------------------------------------------------------------------- metrics
def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    pos = y > 0
    return {
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "MAPE_%": float(np.mean(np.abs(y[pos] - p[pos]) / y[pos]) * 100) if pos.any() else np.nan,
        "WAPE_%": float(np.abs(y - p).sum() / y.sum() * 100),
        "Bias_units": float(np.mean(p - y)),
        "n_rows": int(len(y)),
        "n_zero_actual_excluded_from_MAPE": int((~pos).sum()),
    }


def score_table(preds: pd.DataFrame, models: list[str], truth: pd.DataFrame, by_horizon=True) -> pd.DataFrame:
    """One row per (model, horizon) plus a pooled 'all' row over horizons 1-3."""
    d = preds.merge(truth, on=["product_id", "target_idx", "horizon"], how="inner")
    d = d[d["is_imputed_target"] == 0]
    rows = []
    for m in models:
        for h in sorted(d["horizon"].unique()):
            s = d[d["horizon"] == h]
            rows.append({"model": m, "horizon": str(h), **metrics(s["y_true"], s[m])})
        rows.append({"model": m, "horizon": "all", **metrics(d["y_true"], d[m])})
    return pd.DataFrame(rows)


class HoldoutLock:
    """Makes 'touch the holdout once' enforceable: a second use raises."""

    def __init__(self):
        self._used = False

    def use(self):
        if self._used:
            raise RuntimeError("Holdout already used - it may only be evaluated once.")
        self._used = True


# --------------------------------------------------------------------------- charts
def _style(ax, title, ylabel=None, xlabel=None):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors="#52514e", labelsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color="#52514e", fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color="#52514e", fontsize=9)


def _save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_actual_vs_pred(panel, preds, truth, best: str, baseline: str, extra_pred_col=None, name="actual_vs_predicted.png"):
    """6 sample products (2 high / 2 mid / 2 low volume), holdout block, 1-week-ahead forecasts."""
    d = preds[preds["horizon"] == 1].merge(truth[truth["horizon"] == 1],
                                           on=["product_id", "target_idx", "horizon"])
    vol = d.groupby("product_id")["y_true"].mean().sort_values()
    n = len(vol)
    pick = [vol.index[int(n * q)] for q in (0.92, 0.80, 0.55, 0.45, 0.20, 0.08)]
    import matplotlib.dates as mdates
    fig, axes = plt.subplots(3, 2, figsize=(12, 9.5), sharex=True, layout="constrained")
    for ax, pid in zip(axes.ravel(), pick):
        s = d[d["product_id"] == pid].sort_values("target_idx")
        wk = panel.weeks[s["target_idx"].to_numpy()]
        ax.plot(wk, s["y_true"], color=INK, lw=2, label="Actual")
        ax.plot(wk, s[baseline], color=GREY, lw=1.6, ls="--", label=f"Baseline ({baseline})")
        ax.plot(wk, s[best], color=BLUE, lw=2, label=f"Best model ({best})")
        _style(ax, f"{pid}  (avg {s['y_true'].mean():.0f} units/wk)", "units / week")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Holdout: actual vs 1-week-ahead forecast (sample products, high to low volume)",
                 fontsize=12, color=INK, x=0.01, ha="left", fontweight="bold")
    fig.legend(h, l, loc="outside lower center", ncol=3, frameon=False, fontsize=10)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(C.FIG_DIR / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_model_comparison(table: pd.DataFrame, best: str, baseline: str, name="model_comparison.png"):
    t = table[table["horizon"] == "all"].set_index("model")
    order = list(t.sort_values("RMSE").index)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, metric in zip(axes, ["RMSE", "WAPE_%", "MAPE_%"]):
        vals = t.loc[order, metric]
        colors = [BLUE if m == best else (ORANGE if m == baseline else GREY) for m in order]
        bars = ax.barh(order, vals, color=colors, height=0.6)
        ax.invert_yaxis()
        for b, v in zip(bars, vals):
            ax.text(v, b.get_y() + b.get_height() / 2, f" {v:.1f}", va="center", fontsize=9, color=INK)
        _style(ax, metric.replace("_%", " (%)"))
        ax.grid(axis="x", color=GRID)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, vals.max() * 1.18)
    fig.suptitle("Holdout accuracy, horizons 1-3 pooled (lower is better; orange = baseline, blue = best model)",
                 x=0.01, ha="left", fontsize=11, color=INK, fontweight="bold")
    fig.subplots_adjust(top=0.84)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(C.FIG_DIR / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_feature_importance(imps: dict, name="feature_importance.png", top=12):
    fig, axes = plt.subplots(1, len(imps), figsize=(6.2 * len(imps), 5))
    axes = np.atleast_1d(axes)
    for ax, (title, s) in zip(axes, imps.items()):
        s = s.sort_values(ascending=False).head(top)[::-1]
        ax.barh(s.index, s.values, color=BLUE, height=0.65)
        _style(ax, title)
        ax.grid(axis="x", color=GRID)
        ax.grid(axis="y", visible=False)
    _save(fig, name)


def plot_stockouts(res: dict, labels: dict, title: str, name="stockout_comparison.png"):
    """res: scenario -> metrics dict. Three panels: stock-out freq, fill rate, avg inventory."""
    keys = list(res)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4))
    spec = [("stockout_freq_%", "Stock-out frequency (% of product-weeks)"),
            ("fill_rate_%", "Fill rate (%)"),
            ("avg_inventory_units", "Avg end-of-week inventory (units/product)")]
    cols = [ORANGE, BLUE]
    for ax, (k, ptitle) in zip(axes, spec):
        vals = [res[s][k] for s in keys]
        bars = ax.bar([labels[s] for s in keys], vals, color=cols[:len(keys)], width=0.55)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=10, color=INK)
        _style(ax, ptitle)
        ax.set_ylim(0, max(vals) * 1.18)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(C.FIG_DIR / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_weekly_stockouts(weekly: pd.DataFrame, labels: dict, name="stockouts_by_week.png"):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for col, c, ls in zip(weekly.columns, [ORANGE, BLUE], ["--", "-"]):
        ax.plot(weekly.index, weekly[col], color=c, lw=2, ls=ls, label=labels[col])
    _style(ax, "Stock-outs per week across the 60 products (holdout, same buffers)", "products out of stock")
    ax.legend(frameon=False)
    _save(fig, name)
