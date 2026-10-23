"""End-to-end pipeline:  python run_all.py   (add --fast for a quick smoke test)

1 generate synthetic data -> 2 clean -> 3 features + leakage checks -> 4 tune on walk-forward CV
(training block only) -> 5 validation block: pick baseline & best model -> 6 holdout, evaluated
ONCE -> 7 reorder-policy simulation -> 8 charts + result files (+ README results block).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from functools import reduce

import numpy as np
import pandas as pd

from src import config as C
from src import data_gen, data_prep, evaluate as E, features as F, inventory_sim as S, models as M

KEYS = ["product_id", "target_idx", "horizon"]
BASELINES = ["SeasonalNaive", "MovingAvg4"]


def log(msg):
    print(msg, flush=True)


def merge_preds(frames):
    return reduce(lambda a, b: a.merge(b, on=["product_id", "target_idx", "origin_idx", "horizon"]), frames)


def md_table(df: pd.DataFrame, floatfmt="{:.2f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                       for v in r) + " |")
    return "\n".join(lines)


def main(fast: bool):
    t_start = time.time()
    np.random.seed(C.SEED)
    C.OUT_DIR.mkdir(exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    results: dict = {"seed": C.SEED, "fast_mode": fast}

    # ------------------------------------------------------------------ 1-2 data
    log("== 1. Data (SYNTHETIC, fixed seed)")
    data_gen.main()
    raw = pd.read_csv(C.RAW_FILE)
    truth_gt = pd.read_csv(C.GROUND_TRUTH_FILE)
    clean, rep = data_prep.clean(raw)
    qrep = data_prep.quality_vs_truth(clean, truth_gt)
    clean.to_csv(C.CLEAN_FILE, index=False)
    results["data_quality"] = {**rep, **qrep}
    log(f"   cleaning vs injected truth: {qrep}")

    # ------------------------------------------------------------------ 3 features
    log("== 2. Features + leakage checks")
    panel = F.Panel(clean)
    split = F.make_splits(panel.T, C.VAL_WEEKS, C.TEST_WEEKS, C.WARMUP)
    results["splits"] = {k: {"first_week_idx": v[0], "last_week_idx": v[1],
                             "first_week": str(panel.weeks[v[0]].date()),
                             "last_week": str(panel.weeks[v[1]].date())} for k, v in split.items()}
    log(f"   splits: {results['splits']}")
    feats = {h: F.build_features(panel, h) for h in C.HORIZONS}
    results["leakage_checks"] = [F.assert_no_leakage(panel, h, seed=C.SEED) for h in C.HORIZONS]
    log(f"   leakage perturbation test passed: {results['leakage_checks']}")
    truth = pd.concat([f[KEYS + ["y_true", "is_imputed_target"]] for f in feats.values()], ignore_index=True)
    results["n_features"] = len(F.feature_columns())

    # ------------------------------------------------------------------ 4 tuning (train block only)
    log("== 3. Walk-forward tuning on the TRAIN block only (TimeSeriesSplit, gap=h-1)")
    tuned = {}
    for name in M.MODEL_NAMES:
        tuned[name] = M.walk_forward_tune(name, feats, split["train"], fast=fast)
        log(f"   -> {name}: best {tuned[name]['best_params']}  "
            f"CV RMSE {tuned[name]['cv_rmse_mean']:.3f} +/- {tuned[name]['cv_rmse_std']:.3f}")
    results["tuning"] = {k: {kk: vv for kk, vv in v.items() if kk != "all"} for k, v in tuned.items()}

    # ------------------------------------------------------------------ 5 validation block
    log("== 4. Validation block (models trained on TRAIN only)")
    base_frames = {h: M.baseline_forecasts(panel, feats[h]) for h in C.HORIZONS}
    base_all = pd.concat(base_frames.values(), ignore_index=True)

    def block(rng):
        return base_all[(base_all["target_idx"] >= rng[0]) & (base_all["target_idx"] <= rng[1])]

    val_models = [M.fit_predict_block(n, tuned[n]["best_params"], feats, split["val"], fast=fast)
                  for n in M.MODEL_NAMES]
    val_base = block(split["val"])[["product_id", "target_idx", "origin_idx", "horizon"] + BASELINES]
    val_preds = merge_preds([val_base] + val_models)
    val_tab = E.score_table(val_preds, BASELINES + M.MODEL_NAMES, truth)
    val_all = val_tab[val_tab["horizon"] == "all"].set_index("model")["RMSE"]
    baseline = min(BASELINES, key=lambda m: val_all[m])
    best = min(M.MODEL_NAMES, key=lambda m: val_all[m])
    results["selected_on_validation"] = {"baseline": baseline, "best_model": best,
                                         "val_rmse": {k: float(v) for k, v in val_all.items()}}
    log(f"   validation RMSE (h1-3 pooled):\n{val_all.round(3).to_string()}")
    log(f"   baseline = {baseline} | best model = {best}  (chosen on validation, not on holdout)")
    val_tab.to_csv(C.OUT_DIR / "validation_metrics.csv", index=False)

    # ------------------------------------------------------------------ 6 holdout (ONCE)
    log("== 5. Final holdout - evaluated once (models refit on TRAIN+VAL)")
    lock = E.HoldoutLock()
    lock.use()
    kept = {}
    test_models = [M.fit_predict_block(n, tuned[n]["best_params"], feats, split["test"], fast=fast,
                                       keep_models=kept) for n in M.MODEL_NAMES]
    test_base = block(split["test"])[["product_id", "target_idx", "origin_idx", "horizon"] + BASELINES]
    test_preds = merge_preds([test_base] + test_models)
    test_tab = E.score_table(test_preds, BASELINES + M.MODEL_NAMES, truth)
    test_tab.to_csv(C.OUT_DIR / "holdout_metrics.csv", index=False)
    test_preds.to_csv(C.OUT_DIR / "holdout_predictions.csv", index=False)
    pooled = test_tab[test_tab["horizon"] == "all"].set_index("model")
    results["holdout_pooled"] = pooled.reset_index().to_dict(orient="records")
    results["holdout_best_by_rmse"] = str(pooled.loc[M.MODEL_NAMES, "RMSE"].idxmin())
    results["holdout_stronger_baseline_by_rmse"] = str(pooled.loc[BASELINES, "RMSE"].idxmin())
    results["val_selection_matches_holdout_ranking"] = bool(results["holdout_best_by_rmse"] == best)
    log(pooled.round(3).to_string())

    rmse_gain = (1 - pooled.loc[best, "RMSE"] / pooled.loc[baseline, "RMSE"]) * 100
    wape_gain = (1 - pooled.loc[best, "WAPE_%"] / pooled.loc[baseline, "WAPE_%"]) * 100
    results["accuracy_improvement_vs_baseline"] = {"rmse_pct": float(rmse_gain), "wape_pct": float(wape_gain)}
    log(f"   {best} vs {baseline}: RMSE -{rmse_gain:.1f}%  WAPE -{wape_gain:.1f}% (relative)")

    # feature importance (horizon 1 models)
    cols = F.feature_columns()
    rf = kept[("RandomForest", 1)]
    xg = kept[("XGBoost", 1)]
    rd = kept[("Ridge", 1)]
    gain = xg.get_booster().get_score(importance_type="gain")
    imps = {
        "XGBoost (gain), 1-week-ahead": pd.Series({c: gain.get(f"f{i}", 0.0) for i, c in enumerate(cols)}).pipe(lambda s: s / s.sum()),
        "Random Forest (impurity), 1-week-ahead": pd.Series(rf.feature_importances_, index=cols),
    }
    pd.DataFrame(imps).to_csv(C.OUT_DIR / "feature_importance.csv")
    coef = pd.Series(np.abs(rd[-1].coef_), index=cols)
    results["top_features"] = {k: list(v.sort_values(ascending=False).head(5).index) for k, v in imps.items()}

    # ------------------------------------------------------------------ 7 inventory simulation
    log("== 6. Reorder-policy simulation on the holdout")
    t_lo, t_hi = split["test"]
    products = panel.products
    dem = panel.y[:, t_lo:t_hi + 1]
    prc = panel.price[:, t_lo:t_hi + 1]
    init_rate = panel.y[:, t_lo - 4:t_lo].mean(axis=1)          # last 4 pre-holdout weeks
    kw = dict(preds=test_preds, val_preds=val_preds, truth_val=truth, products=products, demand=dem,
              price=prc, t_lo=t_lo, val_block=split["val"], init_rate=init_rate)
    pair = {"baseline": baseline, "best_model": best}

    def compare(common_buffer: bool, sl: float = C.SERVICE_LEVEL, boot: bool = False):
        """Run baseline vs best model through the identical policy.

        common_buffer=True : safety stock sized from the BASELINE's validation error std for
                             both forecasters (a drop-in forecast swap - same buffers).
        common_buffer=False: each forecaster's safety stock uses its own validation error std
                             (a better forecaster earns a smaller buffer).
        """
        r, sm, _ = S.run_policy(pair, sl=sl, common_sigma_from="baseline" if common_buffer else None, **kw)
        b_, m_ = r["baseline"], r["best_model"]
        out = {"service_level": sl, "baseline": b_, "best_model": m_,
               "stockout_events_reduction_%": S.pct_reduction(b_["stockout_events"], m_["stockout_events"]),
               "units_lost_reduction_%": S.pct_reduction(b_["units_lost"], m_["units_lost"]),
               "fill_rate_change_pts": m_["fill_rate_%"] - b_["fill_rate_%"],
               "inventory_change_%": float((m_["avg_inventory_units"] / b_["avg_inventory_units"] - 1) * 100),
               "cost_proxy_change_%": float((m_["total_cost_proxy"] / b_["total_cost_proxy"] - 1) * 100)}
        if boot:
            out["stockout_events_reduction_ci95"] = list(S.bootstrap_reduction(sm["baseline"]["lost"], sm["best_model"]["lost"]))
            out["_sims"] = sm
        return out

    scenA = compare(True, boot=True)      # same buffers
    scenB = compare(False, boot=True)     # each forecaster calibrated to its own error
    sims_A, sims_B = scenA.pop("_sims"), scenB.pop("_sims")
    sens = []
    for sl in (0.90, 0.95, 0.98):
        a_, b_ = compare(True, sl), compare(False, sl)
        sens.append({"service_level": sl,
                     "A_base_events": a_["baseline"]["stockout_events"], "A_model_events": a_["best_model"]["stockout_events"],
                     "A_reduction_%": a_["stockout_events_reduction_%"], "A_inventory_change_%": a_["inventory_change_%"],
                     "B_base_events": b_["baseline"]["stockout_events"], "B_model_events": b_["best_model"]["stockout_events"],
                     "B_reduction_%": b_["stockout_events_reduction_%"], "B_inventory_change_%": b_["inventory_change_%"]})
    other = {}
    for m in M.MODEL_NAMES:
        r, _, _ = S.run_policy({"baseline": baseline, m: m}, common_sigma_from="baseline", **kw)
        other[m] = r[m]
    results["inventory"] = {"A_same_buffers": scenA, "B_own_error_buffers": scenB,
                            "service_level_sensitivity": sens, "all_models_scenario_A": other,
                            "baseline_name": baseline, "best_name": best,
                            "assumptions": {"lead_time_weeks": C.LEAD_TIME, "review_period_weeks": C.REVIEW_PERIOD,
                                            "service_level": C.SERVICE_LEVEL,
                                            "holding_rate_annual": C.HOLDING_RATE_ANNUAL, "lost_margin": C.LOST_MARGIN}}
    log(f"   A (same buffers): {scenA}\n   B (own-error buffers): {scenB}")

    # ------------------------------------------------------------------ 8 charts + files
    log("== 7. Charts and result files")
    E.plot_actual_vs_pred(panel, test_preds, truth, best, baseline)
    E.plot_model_comparison(test_tab, best, baseline)
    E.plot_feature_importance(imps)
    lab = {"baseline": f"Baseline\n({baseline})", "best_model": f"Best model\n({best})"}
    E.plot_stockouts({"baseline": scenA["baseline"], "best_model": scenA["best_model"]}, lab,
                     "Scenario A - same safety-stock buffers", "stockout_comparison.png")
    E.plot_stockouts({"baseline": scenB["baseline"], "best_model": scenB["best_model"]}, lab,
                     "Scenario B - buffers sized from each forecaster's own error", "stockout_comparison_own_buffers.png")
    weekly = pd.DataFrame({"baseline": (sims_A["baseline"]["lost"] > 0).sum(axis=0),
                           "best_model": (sims_A["best_model"]["lost"] > 0).sum(axis=0)},
                          index=panel.weeks[t_lo + C.LEAD_TIME:t_hi + 1])
    E.plot_weekly_stockouts(weekly, {"baseline": f"Baseline ({baseline})", "best_model": f"Best model ({best})"})
    pd.DataFrame(sens).to_csv(C.OUT_DIR / "inventory_service_level_sensitivity.csv", index=False)
    pd.DataFrame({"A_baseline": scenA["baseline"], "A_best": scenA["best_model"], "B_baseline": scenB["baseline"],
                  "B_best": scenB["best_model"]}).T.to_csv(C.OUT_DIR / "inventory_simulation_summary.csv")

    results["runtime_seconds"] = round(time.time() - t_start, 1)
    with open(C.OUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    write_summary(results, test_tab, pooled, baseline, best)
    log(f"== Done in {results['runtime_seconds']}s.  Results: {C.OUT_DIR}")


def write_summary(res, test_tab, pooled, baseline, best):
    """Writes outputs/results_summary.md and refreshes the README results block (numbers from this run)."""
    p = pooled
    inv = res["inventory"]
    A, B = inv["A_same_buffers"], inv["B_own_error_buffers"]
    order = BASELINES + M.MODEL_NAMES
    comp = p.loc[order, ["RMSE", "MAPE_%", "WAPE_%", "Bias_units"]].reset_index()
    comp["role"] = ["baseline (stronger on validation)" if m == baseline else "baseline" if m in BASELINES
                    else "best model (by validation)" if m == best else "model" for m in comp["model"]]
    perh = test_tab[test_tab["horizon"] != "all"].pivot(index="model", columns="horizon", values="RMSE").loc[order]
    perh.columns = [f"RMSE h={c}" for c in perh.columns]
    zeros = int(p.loc[best, "n_zero_actual_excluded_from_MAPE"])
    n = int(p.loc[best, "n_rows"])

    def inv_tab(sc):
        rows = [("Stock-out events (product-weeks)", "stockout_events"), ("Stock-out frequency (%)", "stockout_freq_%"),
                ("Fill rate (%)", "fill_rate_%"), ("Units lost", "units_lost"),
                ("Avg end-of-week inventory (units/product)", "avg_inventory_units"),
                ("Cost proxy", "total_cost_proxy")]
        return md_table(pd.DataFrame({"": [r[0] for r in rows],
                                      f"Baseline ({baseline})": [sc["baseline"][r[1]] for r in rows],
                                      f"Best model ({best})": [sc["best_model"][r[1]] for r in rows]}), "{:.2f}")

    def verdict(sc):
        ci = sc["stockout_events_reduction_ci95"]
        d = "reduction" if sc["stockout_events_reduction_%"] >= 0 else "INCREASE"
        invw = "higher" if sc["inventory_change_%"] > 0 else "lower"
        return (f"**Stock-out events: {abs(sc['stockout_events_reduction_%']):.1f}% {d}** vs baseline "
                f"(95% bootstrap CI over products for the reduction: {ci[0]:.1f}% to {ci[1]:.1f}%). "
                f"Units lost: {sc['units_lost_reduction_%']:.1f}% reduction. Fill rate: {sc['fill_rate_change_pts']:+.2f} pts. "
                f"**Average inventory is {abs(sc['inventory_change_%']):.1f}% {invw}.** Cost proxy: {sc['cost_proxy_change_%']:+.1f}%.")

    others = pd.DataFrame([{"forecaster": k, "stockout_events": v["stockout_events"], "fill_rate_%": v["fill_rate_%"],
                            "avg_inventory_units": v["avg_inventory_units"]} for k, v in inv["all_models_scenario_A"].items()])
    base_row = pd.DataFrame([{"forecaster": baseline + " (baseline)", "stockout_events": A["baseline"]["stockout_events"],
                              "fill_rate_%": A["baseline"]["fill_rate_%"], "avg_inventory_units": A["baseline"]["avg_inventory_units"]}])
    others = pd.concat([base_row, others], ignore_index=True)
    sens_tab = pd.DataFrame(inv["service_level_sensitivity"])

    block = f"""<!-- RESULTS:START (auto-generated by run_all.py - do not edit by hand) -->
### Forecast accuracy - final holdout ({res['splits']['test']['first_week']} to {res['splits']['test']['last_week']}, evaluated once)

Pooled over 1-, 2- and 3-week-ahead forecasts ({n:,} product-week forecasts). Baseline = **{baseline}** (the stronger of the
two baselines on the validation block); best model = **{best}** (lowest validation RMSE). Units are items per product-week.

{md_table(comp, "{:.2f}")}

Per-horizon RMSE:

{md_table(perh.reset_index(), "{:.2f}")}

* **{best} vs {baseline}: RMSE {rmse_txt(res['accuracy_improvement_vs_baseline']['rmse_pct'])}, WAPE {rmse_txt(res['accuracy_improvement_vs_baseline']['wape_pct'])}** (relative).
* MAPE is computed only where actual demand > 0; {zeros} zero-demand forecasts were excluded from MAPE (they stay in RMSE and WAPE).
* The model chosen on validation {"is also" if res['val_selection_matches_holdout_ranking'] else "is NOT"} the best model on the holdout by RMSE (holdout best: {res['holdout_best_by_rmse']}).

### Reorder simulation - holdout, {C.LEAD_TIME}-week lead time, {int(C.SERVICE_LEVEL*100)}% target service level, identical starting stock

Two ways of running the *same* policy are reported because they answer different questions.

**Scenario A - same buffers (drop-in forecast swap).** Safety stock is sized from the baseline's validation error std and applied to both
forecasters, so only the forecast changes.

{inv_tab(A)}

{verdict(A)}

**Scenario B - buffers sized from each forecaster's own error.** A more accurate forecast earns a smaller safety stock.

{inv_tab(B)}

{verdict(B)}

All models through scenario A:

{md_table(others, "{:.2f}")}

Sensitivity to the target service level (A = same buffers, B = own-error buffers; reduction = % fewer stock-out events, inventory change = % change in average inventory):

{md_table(sens_tab, "{:.2f}")}
<!-- RESULTS:END -->"""
    (C.OUT_DIR / "results_summary.md").write_text(block + "\n", encoding="utf-8")
    readme = C.ROOT / "README.md"
    if readme.exists():
        txt = readme.read_text(encoding="utf-8")
        readme.write_text(re.sub(r"<!-- RESULTS:START.*?<!-- RESULTS:END -->", lambda m: block, txt, flags=re.S),
                          encoding="utf-8")

    bullets = f"""* Trained and compared Ridge (linear), Random Forest, and XGBoost to forecast weekly product demand; best model ({best}) achieved RMSE {p.loc[best, 'RMSE']:.2f} and MAPE {p.loc[best, 'MAPE_%']:.1f}% vs a {baseline} baseline (RMSE {p.loc[baseline, 'RMSE']:.2f}, MAPE {p.loc[baseline, 'MAPE_%']:.1f}%).
* Fed forecasts into reorder logic: with identical safety-stock buffers, projected stock-out events fell {A['stockout_events_reduction_%']:.1f}% (average inventory {A['inventory_change_%']:+.1f}%); with buffers re-sized to each forecast's own error, stock-out frequency was {B['stockout_events_reduction_%']:+.1f}% different and average inventory {B['inventory_change_%']:+.1f}%.
"""
    (C.OUT_DIR / "resume_bullets.md").write_text(bullets, encoding="utf-8")


def rmse_txt(x):
    return f"{abs(x):.1f}% {'lower' if x > 0 else 'HIGHER'}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="smaller grids / fewer trees (smoke test)")
    a = ap.parse_args()
    main(a.fast)
