"""Baselines, model factories, walk-forward tuning and horizon-wise fit/predict.

All models are *global* (one model for all products) and *direct* (one model per horizon).
Targets are level-normalised (see features.py); predictions are multiplied back by the level.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from . import config as C
from .features import Panel, feature_columns, rows_for_training

MODEL_NAMES = ["Ridge", "RandomForest", "XGBoost"]


# --------------------------------------------------------------------------- baselines
def baseline_forecasts(panel: Panel, feats: pd.DataFrame) -> pd.DataFrame:
    """Add seasonal-naive and moving-average forecasts to a horizon's feature table.

    * seasonal_naive : demand of the same week last year, y[t-52]
    * moving_avg_4   : mean of the last 4 observed weeks at the origin (flat forecast)
    """
    out = feats[["product_id", "target_idx", "origin_idx", "level", "y_true",
                 "is_imputed_target", "horizon"]].copy()
    out["SeasonalNaive"] = (feats["lag_52"] * feats["level"]).to_numpy()
    out["MovingAvg4"] = (feats["roll_mean_4"] * feats["level"]).to_numpy()
    return out


# --------------------------------------------------------------------------- factories
def make_model(name: str, params: dict, seed: int = C.SEED, fast: bool = False):
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=params["alpha"], random_state=seed))
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=80 if fast else 200, random_state=seed,
                                     n_jobs=-1, **params)
    if name == "XGBoost":
        return XGBRegressor(tree_method="hist", subsample=0.8, colsample_bytree=0.8,
                            min_child_weight=5, random_state=seed, n_jobs=2, verbosity=0,
                            **params)
    raise ValueError(name)


def param_grid(name: str, fast: bool = False) -> list[dict]:
    if name == "Ridge":
        grid = {"alpha": [0.1, 1, 10, 100, 1000]}
    elif name == "RandomForest":
        grid = {"max_depth": [6, 12], "min_samples_leaf": [5, 20], "max_features": [0.5, 1.0]}
    else:
        grid = {"max_depth": [3, 5], "learning_rate": [0.03, 0.1], "n_estimators": [200, 400]}
    combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    return combos[::2] if fast else combos


def _fit(model, name, X, y, w):
    if name == "Ridge":
        model.fit(X, y, ridge__sample_weight=w)
    else:
        model.fit(X, y, sample_weight=w)
    return model


def _weights(df):
    # weight by level so the normalised loss approximates the original-scale loss
    # without letting a handful of very large products dominate completely
    return df["level"].to_numpy()


def _predict(model, df, cols):
    return np.clip(model.predict(df[cols].to_numpy()), 0, None) * df["level"].to_numpy()


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


# --------------------------------------------------------------------------- tuning
def walk_forward_tune(name: str, feats_by_h: dict, train_range: tuple, fast: bool = False,
                      verbose: bool = True) -> dict:
    """Pick hyper-parameters by walk-forward CV *inside the training block only*.

    TimeSeriesSplit runs over the sorted unique target weeks (never over shuffled rows),
    with gap = h - 1 so no training label is unknown at the first forecast origin of the
    fold. The validation block and the holdout are never touched here.
    """
    cols = feature_columns()
    lo, hi = train_range
    results = []
    for params in param_grid(name, fast):
        fold_scores = []
        for h, df in feats_by_h.items():
            d = df[(df["target_idx"] >= lo) & (df["target_idx"] <= hi)]
            weeks = np.sort(d["target_idx"].unique())
            tscv = TimeSeriesSplit(n_splits=C.CV_SPLITS, test_size=C.CV_TEST_WEEKS, gap=h - 1)
            for tr_i, te_i in tscv.split(weeks):
                tr, te = d[d["target_idx"].isin(weeks[tr_i])], d[d["target_idx"].isin(weeks[te_i])]
                assert tr["target_idx"].max() + h <= te["target_idx"].min()      # labels known
                m = _fit(make_model(name, params, fast=fast), name, tr[cols].to_numpy(),
                         tr["y_norm"].to_numpy(), _weights(tr))
                te = te[te["is_imputed_target"] == 0]
                fold_scores.append(rmse(te["y_true"], _predict(m, te, cols)))
        results.append({"params": params, "cv_rmse_mean": float(np.mean(fold_scores)),
                        "cv_rmse_std": float(np.std(fold_scores))})
        if verbose:
            print(f"   {name:13s} {params}  CV RMSE={results[-1]['cv_rmse_mean']:.3f}")
    best = min(results, key=lambda r: r["cv_rmse_mean"])
    return {"best_params": best["params"], "cv_rmse_mean": best["cv_rmse_mean"],
            "cv_rmse_std": best["cv_rmse_std"], "all": results}


# --------------------------------------------------------------------------- fit / predict
def fit_predict_block(name: str, params: dict, feats_by_h: dict, eval_range: tuple,
                      fast: bool = False, keep_models: dict | None = None) -> pd.DataFrame:
    """Train on rows whose labels are known at the first forecast, predict the eval block."""
    cols = feature_columns()
    first, last = eval_range
    out = []
    for h, df in feats_by_h.items():
        tr = rows_for_training(df, first, h)
        te = df[(df["target_idx"] >= first) & (df["target_idx"] <= last)].copy()
        m = _fit(make_model(name, params, fast=fast), name, tr[cols].to_numpy(),
                 tr["y_norm"].to_numpy(), _weights(tr))
        te[name] = _predict(m, te, cols)
        out.append(te[["product_id", "target_idx", "origin_idx", "horizon", name]])
        if keep_models is not None:
            keep_models[(name, h)] = m
    return pd.concat(out, ignore_index=True)
