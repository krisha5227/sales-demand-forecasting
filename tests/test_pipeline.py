"""Fast sanity tests:  python -m pytest -q   (or just: python tests/test_pipeline.py)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C, data_gen, data_prep, features as F, evaluate as E, inventory_sim as S  # noqa: E402


def _panel():
    raw, _ = data_gen.generate(seed=1, n_products=10, n_weeks=120)
    clean, _ = data_prep.clean(raw, verbose=False)
    return F.Panel(clean), clean


def test_cleaning_has_no_negatives_or_gaps():
    panel, clean = _panel()
    assert (clean["units"] >= 0).all() and clean["units"].notna().all()
    assert clean.groupby("product_id").size().nunique() == 1          # complete grid


def test_no_feature_leakage_all_horizons():
    panel, _ = _panel()
    for h in C.HORIZONS:
        assert F.assert_no_leakage(panel, h, seed=3)["passed"]


def test_training_rows_are_strictly_before_eval_block():
    panel, _ = _panel()
    for h in C.HORIZONS:
        df = F.build_features(panel, h)
        tr = F.rows_for_training(df, 90, h)
        assert tr["target_idx"].max() + h <= 90


def test_metrics_handle_zero_demand():
    m = E.metrics([0, 10, 20], [1, 9, 22])
    assert m["n_zero_actual_excluded_from_MAPE"] == 1
    assert np.isclose(m["WAPE_%"], 4 / 30 * 100)
    assert np.isclose(m["MAPE_%"], np.mean([0.1, 0.1]) * 100)


def test_holdout_lock_allows_one_use_only():
    lock = E.HoldoutLock()
    lock.use()
    try:
        lock.use()
        raise SystemExit("second use should have raised")
    except RuntimeError:
        pass


def test_inventory_sim_perfect_forecast_no_stockouts():
    P, W, L, R = 1, 8, 2, 1
    demand, price = np.full((P, W), 10.0), np.ones((P, W))
    F_ = np.full((P, W + 1, 3), 10.0)                                  # origins t_lo-1 .. t_lo+W-1
    sim = S.simulate(demand, price, F_, o_min=99, t_lo=100, sigma=np.zeros(P),
                     init_rate=np.array([10.0]), L=L, R=R, z=0.0)
    assert sim["lost"].sum() == 0
    # sigma = 0 and exact forecasts: just-in-time, stock ends every week at 0 with nothing lost
    assert np.allclose(sim["end_stock"][0], 0.0)


def test_inventory_sim_zero_forecast_causes_stockouts():
    P, W = 1, 8
    demand, price = np.full((P, W), 10.0), np.ones((P, W))
    F_ = np.zeros((P, W + 1, 3))
    sim = S.simulate(demand, price, F_, 99, 100, np.zeros(P), np.array([10.0]), 2, 1, 0.0)
    assert sim["lost"].sum() > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
