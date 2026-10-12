from __future__ import annotations
from _paths import method_path, method_config, earth_engine_project

import argparse
import hashlib
import itertools
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = method_path('corn_reference')
QUANTILES = (0.1, 0.5, 0.9)
SEED = 20260913
LGB_THREADS = 4  # User-authorized operational acceleration; candidate grid unchanged.


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pinball(y: np.ndarray, p: np.ndarray, q: float) -> np.ndarray:
    error = y - p
    return np.maximum(q * error, (q - 1.0) * error)


def lgb_model(params: dict, q: float) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(objective="quantile", alpha=q, num_leaves=params["num_leaves"],
                             learning_rate=params["learning_rate"], min_child_samples=params["min_child_samples"],
                             colsample_bytree=params["feature_fraction"], n_estimators=params["n_estimators"],
                             random_state=SEED, n_jobs=LGB_THREADS, verbosity=-1, force_col_wise=True)


def elastic_model(alpha: float, l1_ratio: float) -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("scale", StandardScaler()),
                     ("elastic", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000,
                                            random_state=SEED, precompute=True))])


def novelty(dev: pd.DataFrame, test: pd.DataFrame, weather_features: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    impute = SimpleImputer(strategy="median", keep_empty_features=True)
    scale = StandardScaler()
    x_dev = scale.fit_transform(impute.fit_transform(dev[weather_features]))
    x_test = scale.transform(impute.transform(test[weather_features]))
    pca = PCA(n_components=0.9, svd_solver="full", random_state=SEED)
    z_dev = pca.fit_transform(x_dev)
    z_test = pca.transform(x_test)
    neighbors = NearestNeighbors(n_neighbors=21, metric="euclidean", n_jobs=1).fit(z_dev)
    dev_dist = neighbors.kneighbors(z_dev, return_distance=True)[0][:, 1:21].mean(axis=1)
    test_dist = neighbors.kneighbors(z_test, n_neighbors=20, return_distance=True)[0].mean(axis=1)
    thresholds = np.quantile(dev_dist, [0.25, 0.5, 0.75])
    quartile = np.searchsorted(thresholds, test_dist, side="right") + 1
    meta = {"weather_feature_count": len(weather_features), "pca_components": int(pca.n_components_),
            "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
            "development_quartile_thresholds": thresholds.tolist(), "neighbors": 20}
    return test_dist, quartile, meta


def run(state: str, year: int) -> dict:
    start = time.perf_counter()
    rules = json.loads((ROOT / "00B_TARGET_AND_MODEL_RULE_FREEZE.json").read_text(encoding="utf-8"))
    groups = json.loads((ROOT / "C1L_MODEL_FEATURE_GROUPS.json").read_text(encoding="utf-8"))
    history = pd.read_parquet(ROOT / "06_HISTORY_REFERENCE_PANEL.parquet")
    features = pd.read_parquet(ROOT / "C1L_MODEL_FEATURE_PANEL.parquet").rename(columns={"target_year": "year"})
    target = pd.read_parquet(ROOT / "04_COUNTY_YEAR_TARGET_PANEL.parquet")
    target_main = target.loc[target.year.between(2012, 2024), ["county_fips", "year", "harvested_acres"]]
    data = features.merge(history, on=["state", "county_fips", "year"], how="left", validate="one_to_one")
    data = data.merge(target_main, on=["county_fips", "year"], how="left", validate="one_to_one")
    data = data.loc[data.PRIMARY_CROP_SUPPORT & data.A1_available & data.yield_mg_ha.notna()].copy()
    assert len(data) == 6013
    dev = data.loc[data.state.ne(state) & data.year.lt(year)].sort_values(["year", "state", "county_fips"]).reset_index(drop=True)
    test = data.loc[data.state.eq(state) & data.year.eq(year)].sort_values("county_fips").reset_index(drop=True)
    assert len(dev) > 100 and len(test) >= 20
    assert state not in set(dev.state) and dev.year.max() < year
    inner_year = int(dev.year.max())
    inner_train = dev.loc[dev.year.lt(inner_year)]
    inner_val = dev.loc[dev.year.eq(inner_year)]
    assert len(inner_train) > 100 and len(inner_val) >= 20 and inner_year == year - 1
    assert inner_train.year.max() < inner_val.year.min() < year
    y_train = inner_train.residual_mg_ha.to_numpy(dtype=float)
    y_val = inner_val.residual_mg_ha.to_numpy(dtype=float)
    y_dev = dev.residual_mg_ha.to_numpy(dtype=float)
    pred = test[["state", "county_fips", "year", "yield_bu_acre", "yield_mg_ha", "B_history_mg_ha", "harvested_acres", "valid_composite_fraction", "prior_corn_effective_acres", "STRICT_CROP_SUPPORT", "PRIMARY_CROP_SUPPORT", "A1_available"]].copy()
    pred = pred.rename(columns={"year": "test_year"})
    pred["block_id"] = f"{state}_{year}"
    grids = list(itertools.product(rules["lightgbm"]["num_leaves"], rules["lightgbm"]["learning_rate"],
                                   rules["lightgbm"]["min_child_samples"], rules["lightgbm"]["feature_fraction"],
                                   rules["lightgbm"]["n_estimators"]))
    assert len(grids) == 32
    selections = {}
    for system in ("A0", "AQ", "AS", "A1"):
        system_start = time.perf_counter()
        cols = groups[system]
        x_inner_train = inner_train[cols].replace([np.inf, -np.inf], np.nan)
        x_inner_val = inner_val[cols].replace([np.inf, -np.inf], np.nan)
        x_dev = dev[cols].replace([np.inf, -np.inf], np.nan)
        x_test = test[cols].replace([np.inf, -np.inf], np.nan)
        best_score = float("inf")
        best = None
        # A 700-tree fit has the same deterministic first 300 trees as a
        # separate 300-tree fit. Score both frozen estimator counts without
        # rebuilding the identical prefix; grid order/tie break are unchanged.
        for leaves, rate, min_child, fraction in itertools.product(rules["lightgbm"]["num_leaves"],
                                                                     rules["lightgbm"]["learning_rate"],
                                                                     rules["lightgbm"]["min_child_samples"],
                                                                     rules["lightgbm"]["feature_fraction"]):
            params = {"num_leaves": leaves, "learning_rate": rate, "min_child_samples": min_child,
                      "feature_fraction": fraction, "n_estimators": 700}
            losses = {300: [], 700: []}
            for q in QUANTILES:
                model = lgb_model(params, q)
                model.fit(x_inner_train, y_train)
                for estimators in (300, 700):
                    losses[estimators].append(float(pinball(y_val, model.predict(x_inner_val, num_iteration=estimators), q).mean()))
            for estimators in (300, 700):
                score = float(np.mean(losses[estimators]))
                if score < best_score - 1e-12:
                    best_score, best = score, {**params, "n_estimators": estimators}
        assert best is not None
        selections[system] = {"lightgbm": best, "inner_PB3": best_score, "grid_size": len(grids)}
        for q in QUANTILES:
            model = lgb_model(best, q)
            model.fit(x_dev, y_dev)
            pred[f"{system}_q{int(q*100):02d}_mg_ha"] = test.B_history_mg_ha.to_numpy(dtype=float) + model.predict(x_test)
        en_best_score = float("inf")
        en_best = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for alpha, ratio in itertools.product(rules["elasticnet"]["alpha"], rules["elasticnet"]["l1_ratio"]):
                model = elastic_model(alpha, ratio)
                model.fit(x_inner_train, y_train)
                score = float(np.mean(np.abs(y_val - model.predict(x_inner_val))))
                if score < en_best_score - 1e-12:
                    en_best_score, en_best = score, (alpha, ratio)
            assert en_best is not None
            en_model = elastic_model(*en_best)
            en_model.fit(x_dev, y_dev)
            pred[f"{system}_en_median_mg_ha"] = test.B_history_mg_ha.to_numpy(dtype=float) + en_model.predict(x_test)
        selections[system]["elasticnet"] = {"alpha": en_best[0], "l1_ratio": en_best[1], "inner_MAE": en_best_score, "grid_size": 12}
        print(json.dumps({"block_id": f"{state}_{year}", "system": system,
                          "system_runtime_seconds": time.perf_counter() - system_start,
                          "inner_PB3": best_score, "inner_ElasticNet_MAE": en_best_score}), flush=True)
    weather_cols = groups["A0"]
    novelty_distance, novelty_quartile, novelty_meta = novelty(dev, test, weather_cols)
    pred["climate_novelty_distance"] = novelty_distance
    pred["climate_novelty_quartile"] = novelty_quartile
    assert pred["test_year"].eq(year).all() and pred["state"].eq(state).all()
    assert not pred.duplicated(["county_fips", "test_year"]).any()
    out = ROOT / "folds"
    out.mkdir(exist_ok=True)
    file = out / f"{state}_{year}_predictions.parquet"
    pred.to_parquet(file, index=False, compression="zstd")
    result = {"block_id": f"{state}_{year}", "held_state": state, "test_year": year,
              "status": "COMPLETE", "development_rows": len(dev), "test_rows": len(test),
              "development_states": sorted(dev.state.unique().tolist()), "development_min_year": int(dev.year.min()),
              "development_max_year": int(dev.year.max()), "inner_train_max_year": int(inner_train.year.max()),
              "inner_validation_year": inner_year, "learner_held_state_leakage": False,
              "learner_future_year_leakage": False, "lightgbm_threads": LGB_THREADS,
              "selections": selections,
              "novelty": novelty_meta, "prediction_sha256": sha(file),
              "runtime_seconds": time.perf_counter() - start,
              "completed_utc": datetime.now(timezone.utc).isoformat()}
    (out / f"{state}_{year}_meta.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True, choices=("IA", "IL", "IN", "NE", "KS", "TX"))
    parser.add_argument("--year", required=True, type=int, choices=range(2019, 2025))
    args = parser.parse_args()
    result = run(args.state, args.year)
    print(json.dumps({key: result[key] for key in ("block_id", "status", "development_rows", "test_rows", "runtime_seconds", "prediction_sha256")}), flush=True)
