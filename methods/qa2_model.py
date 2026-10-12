"""QA2 upper-only, cache-backed mechanism fits. Outcome evaluation is separate."""
from __future__ import annotations
from _paths import method_path, method_config, earth_engine_project

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = method_path('qa2')
QA0 = Path(str(method_path('qa0')))
QA1 = Path(str(method_path('qa1')))
ORIGINAL = Path(str(method_path('development')))
LOCK = Path(str(method_path('reconstruction') / 'TV1_RESEARCH_DESIGN_LOCK.json'))
AUTHORITY = Path(str(method_path('reconstruction') / 'TV0_AUTHORITY_AUDIT.json'))
sys.path.insert(0, str(QA1 / "code" / "TV2_runner"))
from tv2_pipeline.loss import action_from_delta, block_weights, equal_block_metric, loss_pair
from tv2_pipeline.models import FoldSafeRepresentation, GRID, weather_optical_residual
from tv2_pipeline.provenance import digest

STATES = ("IA", "IL", "IN", "KS", "NE", "TX")
QS = (10, 50, 90)
ZLEVELS = ("Z0", "Z1", "Z2", "Z3")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def keys(frame: pd.DataFrame) -> list[dict]:
    return [{"state": str(r.state), "year": int(r.year), "county_fips": str(r.county_fips)}
            for r in frame[["state", "year", "county_fips"]].itertuples(index=False)]


class Inputs:
    def __init__(self):
        lock = load(LOCK)
        authority = load(AUTHORITY)
        self.registry = lock
        self.c1 = Path(authority["stages"]["C1L"]["path"])
        self.groups = load(self.c1 / "C1L_MODEL_FEATURE_GROUPS.json")
        self.physical = lock["physical_columns"]
        schema_columns = set(pq.read_schema(self.c1 / "C1L_MODEL_FEATURE_PANEL.parquet").names)
        self.weather = list(dict.fromkeys(self.groups["A0"] + [c for c in ("prior_corn_effective_acres", "prior_crop_support") if c in schema_columns]))
        self.quality = lock["quality_columns"]
        feature_cols = sorted(set(self.groups["A0"] + self.groups["A1"] + self.physical + self.quality + self.weather))
        if not set(feature_cols) <= schema_columns:
            raise ValueError(f"FROZEN_FEATURE_PANEL_MISSING_COLUMNS:{sorted(set(feature_cols)-schema_columns)}")
        base_cols = ["state", "county_fips", "target_year", "PRIMARY_CROP_SUPPORT", "A1_available"]
        f = pd.read_parquet(self.c1 / "C1L_MODEL_FEATURE_PANEL.parquet", columns=base_cols + [c for c in feature_cols if c not in base_cols],
                            filters=[("state", "in", list(STATES)), ("target_year", "<=", 2024)])
        f = f.loc[f.PRIMARY_CROP_SUPPORT & f.A1_available].drop(columns=["PRIMARY_CROP_SUPPORT", "A1_available"]).rename(columns={"target_year": "year"})
        f["county_fips"] = f.county_fips.astype(str).str.zfill(5)
        assert set(f.state) == set(STATES) and int(f.year.max()) == 2024
        self.features = f
        h = pd.read_parquet(self.c1 / "06_HISTORY_REFERENCE_PANEL.parquet", columns=["state", "county_fips", "year", "yield_mg_ha"],
                            filters=[("state", "in", list(STATES)), ("year", "<=", 2024)])
        h["county_fips"] = h.county_fips.astype(str).str.zfill(5)
        self.history = h
        b = pd.read_parquet(self.c1 / "06_HISTORY_REFERENCE_PANEL.parquet",
                            columns=["state", "county_fips", "year", "B_history_mg_ha", "max_history_year"],
                            filters=[("state", "in", list(STATES)), ("year", "<=", 2024)])
        b["county_fips"] = b.county_fips.astype(str).str.zfill(5)
        self.B = b
        p = pd.read_parquet(self.c1 / "11_OUTER_PREDICTIONS.parquet",
                            columns=["state", "county_fips", "test_year"] + [f"{a}_q{q}_mg_ha" for a in ("A0", "A1", "AS") for q in QS],
                            filters=[("state", "in", list(STATES)), ("test_year", ">=", 2019), ("test_year", "<=", 2024)])
        p = p.rename(columns={"test_year": "year"})
        p["county_fips"] = p.county_fips.astype(str).str.zfill(5)
        self.outer_predictions = p
        assert not f.duplicated(["state", "year", "county_fips"]).any()
        assert not h.duplicated(["state", "year", "county_fips"]).any()
        assert not p.duplicated(["state", "year", "county_fips"]).any()

    def feature_block(self, state: str, year: int) -> pd.DataFrame:
        frame = self.features.loc[self.features.state.eq(state) & self.features.year.eq(year)].copy()
        return frame.sort_values("county_fips").reset_index(drop=True)

    def dev_outer(self, state: str, year: int) -> pd.DataFrame:
        result = self.feature_block(state, year).merge(self.outer_predictions.loc[self.outer_predictions.state.eq(state) & self.outer_predictions.year.eq(year)],
                                                        on=["state", "county_fips", "year"], validate="one_to_one")
        assert len(result) > 0
        return result


class CacheIndex:
    def __init__(self, inputs: Inputs):
        self.inputs = inputs
        self.root = ORIGINAL / "source_cache"
        self.index = {}
        self.accessed = {}
        with (QA0 / "QA0_SOURCE_CACHE_KEY_REMAP.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["version"] != "TV2" or row["kind"] != "context":
                    continue
                meta_path = self.root / (row["old_key"] + ".json")
                meta = load(meta_path)
                spec = meta["scientific_spec"]
                key = (spec["geometry"], tuple(spec["excluded_states"]), spec["label_state"], int(spec["label_year"]))
                if key in self.index:
                    raise ValueError(f"AMBIGUOUS_CACHE_CONTEXT:{key}")
                self.index[key] = (row, meta)
        assert len(self.index) == 1320

    def has(self, geometry: str, excluded: tuple[str, ...], state: str, year: int) -> bool:
        return (geometry, tuple(sorted(excluded)), state, year) in self.index

    @lru_cache(maxsize=1024)
    def prediction(self, geometry: str, excluded: tuple[str, ...], state: str, year: int) -> pd.DataFrame:
        key = (geometry, tuple(sorted(excluded)), state, year)
        row, meta = self.index[key]
        old = row["old_key"]
        meta_path, payload_path = self.root / (old + ".json"), self.root / (old + ".bin")
        if sha(meta_path) != row["metadata_sha"] or sha(payload_path) != row["payload_sha"]:
            raise ValueError(f"SOURCE_CACHE_SHA_FAIL:{key}")
        payload = load(payload_path)
        target = self.inputs.feature_block(state, year)
        target = target.merge(self.inputs.B.loc[self.inputs.B.state.eq(state) & self.inputs.B.year.eq(year)],
                              on=["state", "county_fips", "year"], validate="one_to_one")
        target = target.sort_values("county_fips").reset_index(drop=True)
        if digest(keys(target)) != meta["scientific_spec"]["prediction_target_row_key_sha"]:
            raise ValueError(f"SOURCE_CACHE_TARGET_KEY_FAIL:{key}")
        for action in ("A0", "A1"):
            matrix = np.asarray(payload["predictions"][action], float)
            if matrix.shape != (len(target), 3):
                raise ValueError(f"SOURCE_CACHE_PREDICTION_SHAPE:{key}")
            for q, col in zip(QS, matrix.T):
                target[f"{action}_q{q}_mg_ha"] = col
        self.accessed[old] = {"cache_key": old, "metadata_sha256": row["metadata_sha"], "payload_sha256": row["payload_sha"],
                              "geometry": geometry, "excluded_states": list(excluded), "label_state": state, "label_year": year}
        return target

    @lru_cache(maxsize=1024)
    def meta(self, geometry: str, excluded: tuple[str, ...], state: str, year: int) -> pd.DataFrame:
        pred = self.prediction(geometry, excluded, state, year).copy()
        assert year <= 2023
        outcome = self.inputs.history.loc[self.inputs.history.state.eq(state) & self.inputs.history.year.eq(year)]
        pred = pred.merge(outcome, on=["state", "county_fips", "year"], how="left", validate="one_to_one")
        if pred.yield_mg_ha.isna().any():
            raise ValueError(f"HISTORICAL_META_LABEL_MISSING:{state}:{year}")
        a0 = pred[[f"A0_q{q}_mg_ha" for q in QS]].to_numpy(float)
        a1 = pred[[f"A1_q{q}_mg_ha" for q in QS]].to_numpy(float)
        l0, l1, delta = loss_pair(pred.yield_mg_ha.to_numpy(float), a0, a1)
        pred["L0_PB3"], pred["L1_PB3"], pred["delta"] = l0, l1, delta
        return pred


def concat_meta(cache: CacheIndex, geometry: str, excluded: tuple[str, ...], states: tuple[str, ...], years: range) -> pd.DataFrame:
    pieces = [cache.meta(geometry, excluded, state, year) for year in years for state in states]
    frame = pd.concat(pieces, ignore_index=True)
    frame["block_weight"] = block_weights(frame.state.to_numpy(), frame.year.to_numpy())
    return frame


def z3_inputs(train: pd.DataFrame, predict: pd.DataFrame, inputs: Inputs, excluded: set[str], predict_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Registered forward source-only weather-conditioned optical residual."""
    tr, va = train.copy(), predict.copy()
    optical = next((c for c in inputs.physical if c in tr and c in va), None)
    tr["opt_weather_residual_history_only"] = np.nan
    va["opt_weather_residual_history_only"] = np.nan
    if optical is None:
        return tr, va
    for (state, year), indices in tr.groupby(["state", "year"]).groups.items():
        earlier = tr.loc[tr.year.lt(year) & tr.state.ne(state)]
        if len(earlier) >= 5:
            tr.loc[indices, "opt_weather_residual_history_only"] = weather_optical_residual(
                earlier, tr.loc[indices], inputs.weather, optical, label_year=int(year), excluded_states=excluded | {state})
    if len(tr) >= 5:
        va["opt_weather_residual_history_only"] = weather_optical_residual(
            tr, va, inputs.weather, optical, label_year=predict_year, excluded_states=excluded)
    return tr, va


def fit_one(inputs: Inputs, cache: CacheIndex, *, condition: str, outer_state: str, outer_year: int, z: str):
    if condition == "SOURCE_DOMAIN_TEMPORAL":
        partner = STATES[(STATES.index(outer_state) + 1) % len(STATES)]
        excluded = (partner,)
        domain = tuple(s for s in STATES if s != partner)
        outer = cache.prediction("G0", excluded, outer_state, outer_year).copy()
    else:
        excluded = (outer_state,)
        domain = tuple(s for s in STATES if s != outer_state)
        outer = inputs.dev_outer(outer_state, outer_year)
    final = concat_meta(cache, "G0", excluded, domain, range(2016, outer_year))
    folds = []
    for year in range(outer_year - 2, outer_year):
        for state in domain:
            inner_excluded = tuple(sorted(set(excluded) | {state}))
            inner_states = tuple(s for s in domain if s != state)
            tr = concat_meta(cache, "G0", inner_excluded, inner_states, range(2016, year))
            va = cache.meta("G1", inner_excluded, state, year)
            assert tr.year.max() < year and set(tr.state) == set(inner_states)
            folds.append((state, year, tr, va))
    matrices = []
    for state, year, tr, va in folds:
        if z == "Z3":
            tr, va = z3_inputs(tr, va, inputs, set(excluded) | {state}, year)
        transformer = FoldSafeRepresentation("R1", z, inputs.physical, inputs.weather, inputs.quality).fit(tr)
        matrices.append((state, year, transformer.transform(tr), tr.delta.to_numpy(float), tr.block_weight.to_numpy(float),
                         transformer.transform(va), va.L0_PB3.to_numpy(float), va.L1_PB3.to_numpy(float), len(tr), len(va)))
    if z == "Z3":
        final, outer = z3_inputs(final, outer, inputs, set(excluded), outer_year)
    final_transformer = FoldSafeRepresentation("R1", z, inputs.physical, inputs.weather, inputs.quality).fit(final)
    final_x, outer_x = final_transformer.transform(final), final_transformer.transform(outer)
    scores = []
    count = 0
    for order, (leaves, rate, child, trees) in enumerate(GRID):
        fold_scores = []
        for state, year, xt, yd, weights, xv, l0, l1, _, _ in matrices:
            model = lgb.LGBMRegressor(objective="regression", num_leaves=leaves, learning_rate=rate, min_child_samples=child,
                                      colsample_bytree=1., n_estimators=trees, random_state=20260913, n_jobs=1,
                                      verbosity=-1, force_col_wise=True)
            model.fit(xt, yd, sample_weight=weights)
            count += 1
            pred = model.predict(xv)
            selected = np.where(action_from_delta(pred) == 1, l1, l0)
            fold_scores.append(float(np.mean(selected)))  # one state-year validation block
        scores.append((float(np.mean(fold_scores)), order, (leaves, rate, child, trees)))
    selected_score, _, (leaves, rate, child, trees) = min(scores)
    model = lgb.LGBMRegressor(objective="regression", num_leaves=leaves, learning_rate=rate, min_child_samples=child,
                              colsample_bytree=1., n_estimators=trees, random_state=20260913, n_jobs=1,
                              verbosity=-1, force_col_wise=True)
    model.fit(final_x, final.delta.to_numpy(float), sample_weight=final.block_weight.to_numpy(float))
    count += 1
    score = model.predict(outer_x)
    actions = action_from_delta(score)
    result = outer[["state", "year", "county_fips"]].copy()
    result["benefit_score"] = score
    result["action"] = np.where(actions == 1, "A1", "A0")
    if condition == "SOURCE_DOMAIN_TEMPORAL":
        constant = equal_block_metric(final.delta.to_numpy(float), final.state.to_numpy(), final.year.to_numpy())
        result["source_constant_score"] = constant
        result["source_static_action"] = "A1" if constant > 0 else "A0"
    report = {"condition": condition, "block": f"{outer_state}_{outer_year}", "Z": z,
              "source_excluded_partner": partner if condition == "SOURCE_DOMAIN_TEMPORAL" else None,
              "outer_rows": len(result), "final_meta_rows": len(final), "final_meta_blocks": int(final.groupby(["state", "year"]).ngroups),
              "validation_folds": len(folds), "fold_support": [{"state": s, "year": y, "train_rows": ntr, "validation_rows": nva}
                                                    for s, y, _, _, _, _, _, _, ntr, nva in matrices],
              "corrected_final_block_weight_mean": float(final.block_weight.mean()),
              "corrected_final_block_mass_range": [float(final.groupby(["state", "year"]).block_weight.sum().min()),
                                                    float(final.groupby(["state", "year"]).block_weight.sum().max())],
              "grid_size": len(GRID), "upper_fit_count": count, "selected_grid": [leaves, rate, child, trees],
              "source_validation_selected_PB3": selected_score,
              "matrix_feature_count": final_x.shape[1], "exact_registry_columns": list(final_transformer.feature_names_),
              "exact_registry_sha256": digest(list(final_transformer.feature_names_)),
              "Z3_residual_nonmissing_final": int(final.opt_weather_residual_history_only.notna().sum()) if z == "Z3" else None,
              "bottom_source_fits": 0, "new_outcome_rows_opened_for_test": 0}
    return result, report


def run(condition: str, block: str, z: str) -> None:
    state, year_s = block.split("_")
    year = int(year_s)
    if state not in STATES or year not in range(2019, 2025):
        raise ValueError("UNREGISTERED_BLOCK")
    if condition == "SOURCE_DOMAIN_TEMPORAL" and (z != "Z2" or year == 2024):
        raise ValueError("SOURCE_TEMPORAL_2024_NOT_CACHED_OR_Z_NOT_REGISTERED")
    if condition == "UNSEEN_GEOGRAPHY_FUTURE_YEAR" and z not in ZLEVELS:
        raise ValueError("UNKNOWN_Z")
    path = ROOT / "model" / condition / z / block
    checkpoint = path / "checkpoint.json"
    if checkpoint.exists():
        data = load(checkpoint)
        if data.get("status") == "PASS" and sha(path / "predictions.parquet") == data["prediction_sha256"]:
            print(json.dumps({"status": "SKIP_VERIFIED", "condition": condition, "Z": z, "block": block}), flush=True)
            return
        raise ValueError("UNVERIFIED_EXISTING_CHECKPOINT")
    inputs = Inputs()
    cache = CacheIndex(inputs)
    start = time.perf_counter()
    predictions, report = fit_one(inputs, cache, condition=condition, outer_state=state, outer_year=year, z=z)
    path.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(path / "predictions.parquet", index=False)
    report["source_contexts_verified"] = len(cache.accessed)
    report["source_context_keys_sha256"] = digest(sorted(cache.accessed))
    report["wall_seconds"] = time.perf_counter() - start
    dump(path / "model_report.json", report)
    if condition == "UNSEEN_GEOGRAPHY_FUTURE_YEAR" and z == "Z2":
        frozen = load(QA1 / "development36" / block / "model_artifacts" / "frozen_benefit_scores.json")
        reference = {r["county_fips"]: r["benefit_score"] for r in frozen["rows"] if r["policy"] == "TV2_L2_G0_R1"}
        diffs = [abs(float(r.benefit_score) - reference[r.county_fips]) for r in predictions.itertuples(index=False)]
        report["QA1_Z2_max_abs_score_difference"] = max(diffs)
        report["QA1_Z2_exact_action_agreement"] = bool(all((r.benefit_score > 0) == (reference[r.county_fips] > 0) for r in predictions.itertuples(index=False)))
        dump(path / "model_report.json", report)
        if max(diffs) > 1e-8 or not report["QA1_Z2_exact_action_agreement"]:
            raise ValueError(f"QA1_Z2_REPRODUCTION_FAIL:{block}:{max(diffs)}")
    dump(checkpoint, {"status": "PASS", "prediction_sha256": sha(path / "predictions.parquet"),
                      "model_report_sha256": sha(path / "model_report.json"), "bottom_source_fits": 0,
                      "source_contexts_verified": len(cache.accessed), "upper_fit_count": report["upper_fit_count"]})
    print(json.dumps({"status": "PASS", "condition": condition, "Z": z, "block": block,
                      "upper_fits": report["upper_fit_count"], "wall_seconds": report["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=("SOURCE_DOMAIN_TEMPORAL", "UNSEEN_GEOGRAPHY_FUTURE_YEAR"))
    parser.add_argument("--block", required=True)
    parser.add_argument("--Z", required=True)
    args = parser.parse_args()
    run(args.condition, args.block, args.Z)
