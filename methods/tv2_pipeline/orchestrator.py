"""Reusable nested task scheduler. TV1R2 exposes no real execution command.

The public methods accept already role-separated in-memory panels. A future,
separately authorized adapter must provide audited <=2024 source panels and
frozen outer predictions. This module never opens any project outcome file.
"""
from __future__ import annotations
from _paths import method_path, method_config, earth_engine_project

import json
import hashlib
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier

from .artifacts import freeze_actions, freeze_scores
from .cache import CanonicalSourceContextCache, CanonicalSourceFitCache, canonical_source_context_key
from .failure import propagate, task_status
from .loss import action_from_delta, blend_prediction, block_weights, equal_block_metric, pinball, selected_prediction, source_selected_blend, source_selected_static
from .models import FoldSafeRepresentation, GRID, build_meta_labels, fit_comparators, fit_source_pair, weather_optical_residual
from .provenance import ProvenanceGraph, digest, file_sha
from .parallel_source import fit_source_pair_parallel
from .task_graph import plan, registered_development_blocks


def _keys(frame: pd.DataFrame) -> list[dict]:
    return [{"state": str(r.state), "year": int(r.year), "county_fips": str(r.county_fips)} for r in frame[["state", "year", "county_fips"]].itertuples(index=False)]


def _matrix_columns(transformer: FoldSafeRepresentation) -> list[str]:
    """Names of the actual router matrix, including fitted R0 PC columns."""
    if transformer.pca is None:
        return list(transformer.feature_names_)
    return [transformer.feature_names_[i] for i in transformer.other_indices_] + [
        f"R0_physical_PC{i + 1}" for i in range(transformer.pca.n_components_)
    ]


def _nested_fold_provenance(tr: pd.DataFrame, va: pd.DataFrame,
                            transformer: FoldSafeRepresentation,
                            source_parents: list[str], *, state: str, year: int,
                            outer_state: str) -> dict:
    train_keys = sorted(_keys(tr), key=lambda x: (x["state"], x["year"], x["county_fips"]))
    validation_keys = sorted(_keys(va), key=lambda x: (x["state"], x["year"], x["county_fips"]))
    train_states = sorted(set(tr.state.astype(str)))
    if state in train_states or outer_state in train_states:
        raise ValueError("NESTED_TRANSFORM_HELD_STATE_LEAKAGE")
    if int(tr.year.max()) >= year or set(va.state.astype(str)) != {state} or not va.year.eq(year).all():
        raise ValueError("NESTED_TRANSFORM_FUTURE_OR_VALIDATION_KEY_LEAKAGE")
    if not source_parents:
        raise ValueError("NESTED_TRANSFORM_SOURCE_PARENT_MISSING")
    return {
        "validation_keys_sha": digest(validation_keys),
        "validation_keys": validation_keys,
        "transform_train_keys_sha": digest(train_keys),
        "transform_train_keys": train_keys,
        "transform_train_states": train_states,
        "transform_train_year_max": int(tr.year.max()),
        "representation": transformer.representation,
        "transform_object_sha": hashlib.sha256(pickle.dumps(transformer, protocol=5)).hexdigest(),
        "transform_matrix_columns": _matrix_columns(transformer),
        "parent_source_artifact_sha": digest(sorted(source_parents)),
        "parent_source_artifact_shas": sorted(source_parents),
    }


def _frame_sha(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(["state", "year", "county_fips"]).reset_index(drop=True)
    return digest({"columns": list(ordered.columns), "rows": ordered.astype(object).where(pd.notna(ordered), None).values.tolist()})


def _source_record(task: dict, train: pd.DataFrame, b_rows: pd.DataFrame, *, role: str, parents: list[str], Z_columns: list[str] | None = None) -> dict:
    return {"artifact_id": f"{task['purpose']}:{task['predict_state']}:{task['predict_year']}:{task['geometry']}:{role}", "role": role,
            "outer_state": task["outer_state"], "outer_year": task["outer_year"], "validation_state": task["validation_state"], "validation_year": task["validation_year"],
            "label_state": task["label_state"], "label_year": task["label_year"], "allowed_states": list(task["allowed_states"]), "excluded_states": list(task["excluded_states"]),
            "max_train_year": int(train.year.max()), "outcome_role": "historical_source_residual" if role in ("reference_history", "source_fit", "meta_label") else "none",
            "transform_fit_keys": _keys(train), "hyperparameter_selection_keys": [], "B_history_keys": _keys(b_rows),
            "feature_registry_sha": task["feature_registry_sha"], "reference_protocol_sha": task["reference_protocol_sha"], "base_config_sha": task["base_config_sha"],
            "fold_selection_sha": task["fold_selection_sha"], "code_sha": task["code_sha"], "parent_artifact_shas": parents,
            "geometry": task["geometry"], "B_role": "history_only", "Z_columns": Z_columns or [], "source_data_sha": _frame_sha(train)}


class NestedBlockRunner:
    """Build source meta labels, select on nested source folds, freeze outer actions.

    `source_history` is historical source-only (< outer year) and contains
    yield/residual columns. `B_reference` has B_history and max_history_year
    only, never yield/residual. `outer_features` contains no evaluation label.
    """
    def __init__(self, source_features: pd.DataFrame, source_history: pd.DataFrame, B_reference: pd.DataFrame, feature_groups: dict, registry: dict, shas: dict, cache_dir: str | Path, source_file_shas: dict[str, str] | None = None, source_rules: dict | None = None, process_workers: int = 1) -> None:
        self.features = source_features.copy()
        self.history = source_history.copy()
        self.B = B_reference.copy()
        self.groups = feature_groups
        self.registry = registry
        self.shas = shas
        self.source_rules = source_rules
        self.source_file_shas = source_file_shas or {"synthetic_source_features": _frame_sha(source_features), "synthetic_source_history": _frame_sha(source_history), "synthetic_B_reference": _frame_sha(B_reference)}
        self.cache = CanonicalSourceContextCache(cache_dir)
        self.fit_cache = CanonicalSourceFitCache(Path(cache_dir) / "canonical_fits")
        self.process_workers = process_workers
        self.executor = ProcessPoolExecutor(max_workers=process_workers) if process_workers > 1 else None
        self.execution_mode = "FROZEN_C1L_300_700_GRID"
        self.graph = ProvenanceGraph()
        self._preflight()

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

    def _preflight(self) -> None:
        if self.registry.get("schema_version") != "TV1.v1" or len(self.registry.get("factorial", [])) != 8:
            raise ValueError("DESIGN_REGISTRY_DRIFT")
        if self.source_rules is not None and digest(self.source_rules["lightgbm"]) != self.shas["base_config_sha"]:
            raise ValueError("FROZEN_SOURCE_GRID_SHA_MISMATCH")
        if any(int(p.year.max()) > 2024 for p in (self.features, self.history, self.B)):
            raise ValueError("FORBIDDEN_2025_2026_SOURCE_YEAR")
        if any(c in self.B.columns for c in ("residual_mg_ha", "yield_mg_ha", "yield_bu_acre", "PB3")):
            raise ValueError("B_HISTORY_RESIDUAL_MIX")
        if any(c in self.features.columns for c in ("residual_mg_ha", "yield_mg_ha", "yield_bu_acre", "PB3")):
            raise ValueError("OUTCOME_IN_PREDICTOR_PANEL")
        if not {"state", "county_fips", "year", "residual_mg_ha", "yield_mg_ha"} <= set(self.history.columns):
            raise ValueError("SOURCE_HISTORY_ROLE_MISSING")
        if not {"state", "county_fips", "year", "B_history_mg_ha", "max_history_year"} <= set(self.B.columns):
            raise ValueError("B_HISTORY_ROLE_MISSING")
        for panel in (self.features, self.history, self.B):
            if panel.duplicated(["state", "county_fips", "year"]).any():
                raise ValueError("DUPLICATE_SOURCE_COUNTY_YEAR")

    def _context(self, task: dict, *, trees: int, threads: int) -> tuple[pd.DataFrame, str, dict]:
        if self.source_rules is None:
            raise ValueError("FROZEN_SOURCE_SELECTION_RULES_REQUIRED")
        if trees != max(self.source_rules["lightgbm"]["n_estimators"]):
            raise ValueError("SOURCE_EFFECTIVE_TREE_COUNT_RULE_MISMATCH")
        allowed, excluded = set(task["allowed_states"]), set(task["excluded_states"])
        label_state, label_year = task["label_state"], int(task["label_year"])
        fit_features = self.features.loc[self.features.state.isin(allowed) & self.features.year.lt(label_year)]
        fit_history = self.history.loc[self.history.state.isin(allowed) & self.history.year.lt(label_year), ["state", "county_fips", "year", "residual_mg_ha"]]
        train = fit_features.merge(fit_history, on=["state", "county_fips", "year"], validate="one_to_one")
        train = train.loc[train.residual_mg_ha.notna()].sort_values(["state", "year", "county_fips"]).reset_index(drop=True)
        target = self.features.loc[self.features.state.eq(label_state) & self.features.year.eq(label_year)].copy()
        target_B = self.B.loc[self.B.state.eq(label_state) & self.B.year.eq(label_year), ["state", "county_fips", "year", "B_history_mg_ha", "max_history_year"]]
        target = target.merge(target_B, on=["state", "county_fips", "year"], validate="one_to_one")
        target = target.sort_values(["state", "year", "county_fips"]).reset_index(drop=True)
        if train.empty or target.empty:
            raise RuntimeError("NOT_ESTIMABLE_NO_SOURCE_OR_TARGET_ROWS")
        if target.B_history_mg_ha.isna().any() or target.max_history_year.isna().any():
            raise RuntimeError("NOT_ESTIMABLE_MISSING_PRIOR_B_REFERENCE")
        if set(train.state) & excluded or train.year.max() >= label_year or target.max_history_year.ge(label_year).any():
            raise ValueError("CONTEXT_SOURCE_LEAKAGE")
        if task["geometry"] == "G1" and label_state not in excluded:
            raise ValueError("G1_LABEL_NOT_EXCLUDED")
        train_B = self.B.loc[self.B.state.isin(allowed) & self.B.year.lt(label_year)]
        if train_B.max_history_year.ge(train_B.year).any():
            raise ValueError("B_HISTORY_NOT_PRIOR")
        input_record = _source_record(task, train, train_B, role="reference_history", parents=[])
        input_record["source_file_shas"] = self.source_file_shas
        input_record.update({"effective_tree_count": trees, "execution_mode": self.execution_mode, "threads": threads})
        input_sha = self.graph.add(input_record)
        canonical_spec = {
            "exact_training_row_key_sha": digest(_keys(train)),
            "exact_validation_row_key_sha": digest(_keys(train.loc[train.year.eq(train.year.max())])),
            "prediction_target_row_key_sha": digest(_keys(target)),
            "training_values_and_keys_sha": _frame_sha(train),
            "target_predictor_and_B_sha": _frame_sha(target),
            "source_files_sha": digest(self.source_file_shas),
            "allowed_states": sorted(allowed), "excluded_states": sorted(excluded),
            "label_state": label_state, "label_year": label_year,
            "geometry": task["geometry"], "actions": ["A0", "A1"],
            "quantiles": [0.1, 0.5, 0.9],
            "feature_registry_sha": task["feature_registry_sha"],
            "reference_protocol_sha": task["reference_protocol_sha"],
            "base_config_sha": task["base_config_sha"],
            "source_grid_sha": digest(self.source_rules["lightgbm"]),
            "code_sha": task["code_sha"],
            "preprocessing_protocol_sha": digest("C1L_NATIVE_NAN_PRIMARY_SUPPORT_A1_AVAILABLE_HISTORY_B_PRIOR_v1"),
            "effective_tree_count": trees, "threads": threads,
            "execution_mode": self.execution_mode,
        }
        key = canonical_source_context_key(canonical_spec)
        try:
            cached = json.loads(self.cache.get(canonical_spec).decode())
            pred = {name: np.asarray(cached["predictions"][name], float) for name in ("A0", "A1")}
            selections = cached["source_selections"]
            hit = True
            fit_count = 0
            fit_cache_hits = 0
        except FileNotFoundError:
            identity = {"feature_registry_sha": task["feature_registry_sha"], "reference_protocol_sha": task["reference_protocol_sha"], "code_sha": task["code_sha"], "source_file_shas": self.source_file_shas, "preprocessing_protocol_sha": canonical_spec["preprocessing_protocol_sha"]}
            target_predictors = target.drop(columns=["B_history_mg_ha", "max_history_year"])
            if self.executor is None:
                pred_residual, selections, fit_count = fit_source_pair(train, target_predictors, self.groups, label_year=label_year, excluded_states=excluded, held_label_state=label_state if task["geometry"] == "G1" else None, n_trees=trees, threads=threads, source_rules=self.source_rules, source_fit_cache=self.fit_cache, cache_identity=identity)
            else:
                pred_residual, selections, fit_count = fit_source_pair_parallel(train, target_predictors, self.groups, label_year=label_year, excluded_states=excluded, held_label_state=label_state if task["geometry"] == "G1" else None, source_rules=self.source_rules, source_fit_cache=self.fit_cache, cache_identity=identity, executor=self.executor)
            rules = self.source_rules["lightgbm"]
            expected_fits = 6 * (len(rules["num_leaves"]) * len(rules["learning_rate"]) * len(rules["min_child_samples"]) * len(rules["feature_fraction"]) + 1)
            fit_cache_hits = expected_fits - fit_count
            if fit_cache_hits < 0:
                raise ValueError("SOURCE_FIT_CACHE_HIT_COUNT_INVALID")
            B = target.B_history_mg_ha.to_numpy(float)[:, None]
            pred = {name: values + B for name, values in pred_residual.items()}
            self.cache.put(canonical_spec, json.dumps({"predictions": {name: values.tolist() for name, values in pred.items()}, "source_selections": selections}, sort_keys=True).encode())
            hit = False
        fit_record = _source_record(task, train, train_B, role="source_fit", parents=[input_sha])
        fit_record["source_file_shas"] = self.source_file_shas
        fit_record.update({"effective_tree_count": trees, "execution_mode": self.execution_mode, "threads": threads, "hyperparameter_selection_keys": _keys(train.loc[train.year.eq(train.year.max())]), "selected_source_configs": selections, "canonical_source_context_key": key, "source_context_cache_hit": hit, "source_fit_cache_hits": fit_cache_hits})
        fit_sha = self.graph.add(fit_record)
        historical_label = self.history.loc[self.history.state.eq(label_state) & self.history.year.eq(label_year), ["state", "county_fips", "year", "yield_mg_ha"]]
        if label_state == task["outer_state"] and label_year == task["outer_year"]:
            raise ValueError("OUTER_EVALUATION_LABEL_FORBIDDEN")
        meta = target.merge(historical_label, on=["state", "county_fips", "year"], validate="one_to_one")
        if len(meta) != len(target):
            raise RuntimeError("NOT_ESTIMABLE_HISTORICAL_META_LABEL_MISSING")
        if meta.yield_mg_ha.isna().any():
            raise RuntimeError("NOT_ESTIMABLE_HISTORICAL_META_LABEL_NULL")
        labels = build_meta_labels(meta.yield_mg_ha.to_numpy(float), pred["A0"], pred["A1"], meta.state.to_numpy(), meta.year.to_numpy())
        for name in ("A0", "A1"):
            for q, values in zip((10, 50, 90), pred[name].T):
                meta[f"{name}_q{q}_mg_ha"] = values
        for col in ("L0_PB3", "L1_PB3", "delta", "winner", "block_weight"):
            meta[col] = labels[col].to_numpy()
        label_record = _source_record(task, train, train_B, role="meta_label", parents=[fit_sha])
        label_record["source_file_shas"] = self.source_file_shas
        label_record.update({"effective_tree_count": trees, "execution_mode": self.execution_mode, "threads": threads})
        label_sha = self.graph.add(label_record)
        return meta, label_sha, {"cache_key": key, "cache_hit": hit, "source_fit_cache_hits": fit_cache_hits, "train_rows": len(train), "label_rows": len(meta), "fit_models": fit_count, "effective_tree_count": trees, "execution_mode": self.execution_mode, "threads": threads, "source_selection": selections}

    def _matched_g0_rows(self, g0: pd.DataFrame, g1: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Frozen-seed, label-year-stratified source-only G0 size sensitivity."""
        selected = []
        counts = {}
        for year, rows in g0.groupby("year", sort=True):
            rows = rows.sort_values(["state", "county_fips"]).reset_index(drop=True)
            target = int(g1.year.eq(year).sum())
            take = min(len(rows), target)
            rng = np.random.default_rng(int(self.registry["geometry"]["matched_size_control_seed"]) + int(year))
            idx = np.sort(rng.choice(np.arange(len(rows)), size=take, replace=False))
            selected.append(rows.iloc[idx])
            counts[str(year)] = {"G0_available": len(rows), "G1_target": target, "taken": take,
                                 "G0_less_than_G1": len(rows) < target}
        if not selected:
            raise ValueError("MATCHED_G0_NO_SOURCE_ROWS")
        result = pd.concat(selected, ignore_index=True).sort_values(["state", "year", "county_fips"]).reset_index(drop=True)
        return result, {"seed": self.registry["geometry"]["matched_size_control_seed"],
                        "stratification": "label_year", "counts": counts,
                        "selected_keys_sha": digest(sorted(_keys(result), key=lambda x: (x["state"], x["year"], x["county_fips"]))) }

    def _nested_strategy(self, task_rows: dict[str, pd.DataFrame], task_shas: dict[str, str], tasks: list[dict], outer_features: pd.DataFrame, *, geometry: str, representation: str, objective: str, selection_metric: str, tree_limit: int | None, matched_size: bool = False) -> tuple[np.ndarray, dict]:
        if matched_size and (geometry, representation, objective, selection_metric) != ("G0", "R1", "L2", "SELECTED_PB3"):
            raise ValueError("UNREGISTERED_MATCHED_SIZE_SENSITIVITY")
        final = pd.concat([task_rows[t["cache_key"]] for t in tasks if t["purpose"] == "final_meta" and t["geometry"] == geometry], ignore_index=True)
        matched_provenance = {}
        if matched_size:
            g1_final = pd.concat([task_rows[t["cache_key"]] for t in tasks if t["purpose"] == "final_meta" and t["geometry"] == "G1"], ignore_index=True)
            final, matched_provenance["final"] = self._matched_g0_rows(final, g1_final)
        final["block_weight"] = block_weights(final.state.to_numpy(), final.year.to_numpy())
        folds = sorted({(t["validation_state"], t["validation_year"]) for t in tasks if t["purpose"] == "router_validation_pair"})
        physical = self.registry["physical_columns"]
        weather = list(dict.fromkeys(self.groups["A0"] + [c for c in ("prior_corn_effective_acres", "prior_crop_support") if c in self.features]))
        quality = self.registry["quality_columns"]
        scores = []
        fold_provenance = {}
        outer_state = str(outer_features.state.iloc[0])
        for order, (leaves, rate, child, trees) in enumerate(GRID):
            fold_scores = []
            for state, year in folds:
                tr = pd.concat([task_rows[t["cache_key"]] for t in tasks if t["purpose"] == "router_train_meta" and t["geometry"] == geometry and t["validation_state"] == state and t["validation_year"] == year], ignore_index=True)
                va = pd.concat([task_rows[t["cache_key"]] for t in tasks if t["purpose"] == "router_validation_pair" and t["validation_state"] == state and t["validation_year"] == year], ignore_index=True)
                if matched_size:
                    g1_tr = pd.concat([task_rows[t["cache_key"]] for t in tasks if t["purpose"] == "router_train_meta" and t["geometry"] == "G1" and t["validation_state"] == state and t["validation_year"] == year], ignore_index=True)
                    tr, matched_provenance[f"{state}_{year}"] = self._matched_g0_rows(tr, g1_tr)
                if tr.empty or va.empty:
                    raise ValueError("NESTED_VALIDATION_NOT_ESTIMABLE")
                tr["block_weight"] = block_weights(tr.state.to_numpy(), tr.year.to_numpy())
                transformer = FoldSafeRepresentation(representation, "Z2", physical, weather, quality).fit(tr)
                if order == 0:
                    parent_shas = [task_shas[t["cache_key"]] for t in tasks if t["purpose"] == "router_train_meta" and t["geometry"] == geometry and t["validation_state"] == state and t["validation_year"] == year]
                    fold_provenance[f"{state}_{year}"] = _nested_fold_provenance(tr, va, transformer, parent_shas, state=state, year=year, outer_state=outer_state)
                xt, xv = transformer.transform(tr), transformer.transform(va)
                model = lgb.LGBMRegressor(objective="regression_l1" if objective == "L1" else "regression", num_leaves=leaves, learning_rate=rate, min_child_samples=child, colsample_bytree=1., n_estimators=min(trees, tree_limit) if tree_limit else trees, random_state=20260913, n_jobs=1, verbosity=-1, force_col_wise=True)
                model.fit(xt, tr.delta.to_numpy(), sample_weight=tr.block_weight.to_numpy())
                pred = model.predict(xv)
                if selection_metric == "MAE":
                    score = equal_block_metric(np.abs(va.delta.to_numpy() - pred), va.state.to_numpy(), va.year.to_numpy())
                else:
                    chosen = np.where(action_from_delta(pred) == 1, va.L1_PB3, va.L0_PB3)
                    score = equal_block_metric(chosen, va.state.to_numpy(), va.year.to_numpy())
                fold_scores.append(score)
            scores.append((float(np.mean(fold_scores)), order, (leaves, rate, child, trees)))
        _, _, (leaves, rate, child, trees) = min(scores)
        transformer = FoldSafeRepresentation(representation, "Z2", physical, weather, quality).fit(final)
        model = lgb.LGBMRegressor(objective="regression_l1" if objective == "L1" else "regression", num_leaves=leaves, learning_rate=rate, min_child_samples=child, colsample_bytree=1., n_estimators=min(trees, tree_limit) if tree_limit else trees, random_state=20260913, n_jobs=1, verbosity=-1, force_col_wise=True)
        model.fit(transformer.transform(final), final.delta.to_numpy(), sample_weight=final.block_weight.to_numpy())
        return model.predict(transformer.transform(outer_features)), {"selected_grid": [leaves, rate, child, trees], "effective_router_tree_count": min(trees, tree_limit) if tree_limit else trees, "nested_validation_folds": len(folds), "source_validation_score": min(scores)[0], "fit_objective": objective, "selection_metric": selection_metric, "representation": representation, "ladder": "Z2", "matrix_columns": _matrix_columns(transformer), "nested_transform_provenance": fold_provenance, "matched_size_provenance": matched_provenance, "matched_final_training_keys": _keys(final) if matched_size else None}

    def _nested_comparators(self, task_rows: dict[str, pd.DataFrame], task_shas: dict[str, str], tasks: list[dict], outer_features: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict]:
        final = pd.concat([task_rows[t["cache_key"]] for t in tasks if t["purpose"] == "final_meta" and t["geometry"] == "G1"], ignore_index=True)
        final["block_weight"] = block_weights(final.state.to_numpy(), final.year.to_numpy())
        folds = sorted({(t["validation_state"], t["validation_year"]) for t in tasks if t["purpose"] == "router_validation_pair"})
        weather = list(dict.fromkeys(self.groups["A0"] + [c for c in ("prior_corn_effective_acres", "prior_crop_support") if c in self.features]))

        def candidate(family: str, parameter: float, x: np.ndarray, delta: np.ndarray, weight: np.ndarray):
            if family == "ridge":
                return Ridge(alpha=parameter).fit(x, delta, sample_weight=weight)
            classes = action_from_delta(delta)
            if len(np.unique(classes)) == 1:
                return int(classes[0])
            if family == "weighted_logit":
                return LogisticRegression(C=parameter, random_state=20260913, max_iter=1000).fit(x, classes, sample_weight=weight * np.abs(delta))
            return DecisionTreeClassifier(max_depth=2, random_state=20260913).fit(x, classes, sample_weight=weight * np.abs(delta))

        def predict_action(family: str, model, x: np.ndarray) -> np.ndarray:
            return np.full(len(x), model, dtype=np.int8) if isinstance(model, int) else action_from_delta(model.predict(x)) if family == "ridge" else model.predict(x)

        models, selection = {}, {}
        outer_state = str(outer_features.state.iloc[0])
        for family, parameters in (("ridge", (.1, 1., 10., 100.)), ("weighted_logit", (.1, 1., 10.)), ("depth2_tree", (2.,))):
            scores = []
            fold_provenance = {}
            for order, parameter in enumerate(parameters):
                fold_scores = []
                for state, year in folds:
                    tr = pd.concat([task_rows[t["cache_key"]] for t in tasks if t["purpose"] == "router_train_meta" and t["geometry"] == "G1" and t["validation_state"] == state and t["validation_year"] == year], ignore_index=True)
                    va = pd.concat([task_rows[t["cache_key"]] for t in tasks if t["purpose"] == "router_validation_pair" and t["validation_state"] == state and t["validation_year"] == year], ignore_index=True)
                    tr["block_weight"] = block_weights(tr.state.to_numpy(), tr.year.to_numpy())
                    transformer = FoldSafeRepresentation("R1", "Z2", self.registry["physical_columns"], weather, self.registry["quality_columns"]).fit(tr)
                    if order == 0:
                        parent_shas = [task_shas[t["cache_key"]] for t in tasks if t["purpose"] == "router_train_meta" and t["geometry"] == "G1" and t["validation_state"] == state and t["validation_year"] == year]
                        fold_provenance[f"{state}_{year}"] = _nested_fold_provenance(tr, va, transformer, parent_shas, state=state, year=year, outer_state=outer_state)
                    model = candidate(family, parameter, transformer.transform(tr), tr.delta.to_numpy(), tr.block_weight.to_numpy())
                    action = predict_action(family, model, transformer.transform(va))
                    score = equal_block_metric(np.where(action == 1, va.L1_PB3, va.L0_PB3), va.state.to_numpy(), va.year.to_numpy())
                    fold_scores.append(score)
                scores.append((float(np.mean(fold_scores)), order, parameter))
            score, _, chosen = min(scores)
            transformer = FoldSafeRepresentation("R1", "Z2", self.registry["physical_columns"], weather, self.registry["quality_columns"]).fit(final)
            model = candidate(family, chosen, transformer.transform(final), final.delta.to_numpy(), final.block_weight.to_numpy())
            models[family] = predict_action(family, model, transformer.transform(outer_features))
            selection[family] = {"source_validation_score": score, "selected_parameter": chosen, "nested_validation_folds": len(folds), "representation": "R1", "ladder": "Z2", "matrix_columns": _matrix_columns(transformer), "nested_transform_provenance": fold_provenance}
        return models, selection

    def information_ladder_diagnostic(self, final_G1_meta: pd.DataFrame, outer_features: pd.DataFrame, *, outer_state: str, outer_year: int) -> dict:
        """Construct Z0..Z3 in legal forward source-only geometry; no labels/score."""
        train = final_G1_meta.copy()
        outer = outer_features.copy()
        weather = list(dict.fromkeys(self.groups["A0"] + [c for c in ("prior_corn_effective_acres", "prior_crop_support") if c in self.features]))
        optical = next((c for c in self.registry["physical_columns"] if c in train and c in outer), None)
        train["opt_weather_residual_history_only"] = np.nan
        outer["opt_weather_residual_history_only"] = np.nan
        if optical is not None:
            for (state, year), idx in train.groupby(["state", "year"]).groups.items():
                earlier = train.loc[train.year.lt(year) & train.state.ne(state) & train.state.ne(outer_state)]
                if len(earlier) >= 5:
                    train.loc[idx, "opt_weather_residual_history_only"] = weather_optical_residual(earlier, train.loc[idx], weather, optical, label_year=int(year), excluded_states={outer_state, state})
            if len(train) >= 5:
                outer["opt_weather_residual_history_only"] = weather_optical_residual(train.loc[train.state.ne(outer_state)], outer, weather, optical, label_year=outer_year, excluded_states={outer_state})
        records = {}
        for ladder in ("Z0", "Z1", "Z2", "Z3"):
            for representation in ("R0", "R1"):
                fit = FoldSafeRepresentation(representation, ladder, self.registry["physical_columns"], weather, self.registry["quality_columns"]).fit(train)
                transformed = fit.transform(outer)
                records[f"{ladder}_{representation}"] = {"fit_rows": len(train), "outer_rows": len(outer), "columns": transformed.shape[1], "outer_matrix_sha": digest(np.round(transformed, 8).tolist()), "source_only": True, "outer_label_used": False}
        return {"status": "CONSTRUCTED_NO_SCIENTIFIC_SELECTION", "optical_column": optical, "Z3_forward_history_nonmissing_train": int(train.opt_weather_residual_history_only.notna().sum()), "matrices": records}

    def run_block(self, outer_state: str, outer_year: int, outer_features: pd.DataFrame, out_dir: str | Path, *, synthetic: bool, tree_limit: int | None = None, threads: int = 1, frozen_comparators: dict[str, dict] | None = None) -> dict:
        """All nested contexts and source selection, ending before evaluation."""
        self.execution_mode = "SYNTHETIC_ONLY" if synthetic else "FROZEN_C1L_300_700_GRID"
        if not synthetic and (tree_limit is not None or threads != 1 or self.source_rules is None):
            raise ValueError("REAL_TV2_FROZEN_SOURCE_GRID_OR_THREAD_RULE_DRIFT")
        if any(c in outer_features for c in ("yield_mg_ha", "yield_bu_acre", "residual_mg_ha", "PB3", "benefit_A1_PB3")):
            raise ValueError("OUTER_EVALUATION_LABEL_IN_MODEL_PROCESS")
        if self.history.state.eq(outer_state).any():
            raise ValueError("OUTER_STATE_HISTORY_MUST_BE_ABSENT")
        if outer_features.year.nunique() != 1 or int(outer_features.year.iloc[0]) != outer_year or set(outer_features.state) != {outer_state}:
            raise ValueError("OUTER_FEATURE_KEY_MISMATCH")
        graph = plan(outer_state, outer_year, self.shas)
        tasks = graph["tasks"]
        task_rows, task_shas, costs, failures = {}, {}, [], []
        for task in tasks:
            try:
                meta, art_sha, cost = self._context(task, trees=tree_limit or max(self.source_rules["lightgbm"]["n_estimators"]), threads=threads)
            except RuntimeError as exc:
                failures.append({"cache_key": task["cache_key"], "status": "NOT_ESTIMABLE", "reason": str(exc)})
                continue
            task_rows[task["cache_key"]], task_shas[task["cache_key"]] = meta, art_sha
            costs.append(cost)
        if failures:
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            result = {"status": "NOT_ESTIMABLE", "outer_block": f"{outer_state}_{outer_year}", "required_contexts": len(tasks), "failed_contexts": failures,
                      "outer_label_opened": False, "science_effect_computed": False}
            (out / "synthetic_full_block_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            return result
        actions, predictions, selections, benefit_scores = {}, {}, {}, {}
        a0 = outer_features[[f"A0_q{q}_mg_ha" for q in (10, 50, 90)]].to_numpy(float)
        a1 = outer_features[[f"A1_q{q}_mg_ha" for q in (10, 50, 90)]].to_numpy(float)
        predictions["ALWAYS_A0"], predictions["ALWAYS_A1"] = a0, a1
        actions["ALWAYS_A0"] = np.zeros(len(outer_features), dtype=np.int8)
        actions["ALWAYS_A1"] = np.ones(len(outer_features), dtype=np.int8)
        for objective in ("L1", "L2"):
            for geometry in ("G0", "G1"):
                for representation in ("R0", "R1"):
                    benefit, selection = self._nested_strategy(task_rows, task_shas, tasks, outer_features, geometry=geometry, representation=representation, objective=objective, selection_metric="MAE" if objective == "L1" else "SELECTED_PB3", tree_limit=tree_limit)
                    name = f"TV2_{objective}_{geometry}_{representation}"
                    actions[name] = action_from_delta(benefit)
                    predictions[name] = selected_prediction(a0, a1, actions[name])
                    selections[name] = selection
                    benefit_scores[name] = benefit
        matched_benefit, matched_selection = self._nested_strategy(task_rows, task_shas, tasks, outer_features, geometry="G0", representation="R1", objective="L2", selection_metric="SELECTED_PB3", tree_limit=tree_limit, matched_size=True)
        actions["G0_MATCHED_L2_R1"] = action_from_delta(matched_benefit)
        predictions["G0_MATCHED_L2_R1"] = selected_prediction(a0, a1, actions["G0_MATCHED_L2_R1"])
        selections["G0_MATCHED_L2_R1"] = matched_selection
        final = pd.concat([task_rows[t["cache_key"]] for t in tasks if t["purpose"] == "final_meta" and t["geometry"] == "G1"], ignore_index=True)
        ladder_diagnostic = self.information_ladder_diagnostic(final, outer_features, outer_state=outer_state, outer_year=outer_year)
        g0_final = pd.concat([task_rows[t["cache_key"]] for t in tasks if t["purpose"] == "final_meta" and t["geometry"] == "G0"], ignore_index=True)
        matched_rng = np.random.default_rng(self.registry["geometry"]["matched_size_control_seed"])
        matched_index = np.sort(matched_rng.choice(np.arange(len(g0_final)), size=min(len(g0_final), len(final)), replace=False))
        matched_size_control = {"G0_rows": len(g0_final), "G1_rows": len(final), "matched_rows": len(matched_index), "seed": self.registry["geometry"]["matched_size_control_seed"], "G0_selected_key_sha": digest(_keys(g0_final.iloc[matched_index]))}
        static = source_selected_static(final.delta.to_numpy(), final.state.to_numpy(), final.year.to_numpy())
        actions["SOURCE_SELECTED_STATIC"] = np.full(len(outer_features), static, dtype=np.int8)
        predictions["SOURCE_SELECTED_STATIC"] = selected_prediction(a0, a1, actions["SOURCE_SELECTED_STATIC"])
        lam = source_selected_blend(final.yield_mg_ha.to_numpy(float), final[[f"A0_q{q}_mg_ha" for q in (10,50,90)]].to_numpy(), final[[f"A1_q{q}_mg_ha" for q in (10,50,90)]].to_numpy(), final.state.to_numpy(), final.year.to_numpy())
        predictions["SOURCE_SELECTED_SOFT_BLEND"] = blend_prediction(a0, a1, lam)
        source_meta_keys = sorted(_keys(final), key=lambda x: (x["state"], x["year"], x["county_fips"]))
        source_support = {f"{state}_{year}": int(n) for (state, year), n in final.groupby(["state", "year"]).size().items()}
        static_selection = {"source_meta_label_key_sha": digest(source_meta_keys), "G1_source_state_year_support": source_support,
                            "equal_block_source_delta": equal_block_metric(final.delta.to_numpy(), final.state.to_numpy(), final.year.to_numpy()),
                            "selected_constant_action": "A1" if static else "A0", "action_selection_rule_sha": file_sha(Path(__file__).resolve().parent / "loss.py")}
        lambda_set = (0., .25, .5, .75, 1.)
        blend_source_scores = {str(candidate): equal_block_metric(pinball(final.yield_mg_ha.to_numpy(float), blend_prediction(final[[f"A0_q{q}_mg_ha" for q in (10,50,90)]].to_numpy(), final[[f"A1_q{q}_mg_ha" for q in (10,50,90)]].to_numpy(), candidate)), final.state.to_numpy(), final.year.to_numpy()) for candidate in lambda_set}
        blend_selection = {"source_meta_label_key_sha": digest(source_meta_keys), "G1_source_state_year_support": source_support,
                           "candidate_lambdas": list(lambda_set), "source_only_equal_block_PB3_by_lambda": blend_source_scores,
                           "selected_lambda": lam, "tie_breaking_rule": "minimum PB3 then smaller lambda",
                           "selection_rule_sha": file_sha(Path(__file__).resolve().parent / "loss.py")}
        for name, objective, metric in (("B1_L1_equalblock_MAE", "L1", "MAE"), ("B2_L1_equalblock_selected_PB3", "L1", "SELECTED_PB3"), ("B3_L2_equalblock_selected_PB3", "L2", "SELECTED_PB3")):
            benefit, selection = self._nested_strategy(task_rows, task_shas, tasks, outer_features, geometry="G1", representation="R1", objective=objective, selection_metric=metric, tree_limit=tree_limit)
            actions[name] = action_from_delta(benefit)
            predictions[name] = selected_prediction(a0, a1, actions[name])
            selections[name] = selection
            if name == "B2_L1_equalblock_selected_PB3":
                benefit_scores[name] = benefit
        comps, comp_selection = self._nested_comparators(task_rows, task_shas, tasks, outer_features)
        for family, act in comps.items():
            actions[family] = act
            predictions[family] = selected_prediction(a0, a1, act)
            selections[family] = comp_selection[family]
        frozen_status, frozen_source_provenance = {}, {}
        for name in ("ALWAYS_AS", "FROZEN_C2_FULL_ROUTER"):
            if frozen_comparators and name in frozen_comparators:
                entry = frozen_comparators[name]
                values = np.asarray(entry["prediction"], float)
                expected_role = "frozen_C1L" if name == "ALWAYS_AS" else "frozen_C2"
                if values.shape != a0.shape or digest(values.tolist()) != entry["sha256"] or entry.get("source_role") != expected_role or len(entry.get("source_file_sha", "")) != 64 or len(entry.get("protocol_sha", "")) != 64 or (name == "FROZEN_C2_FULL_ROUTER" and entry.get("router_id") != "RF"):
                    raise ValueError(f"FROZEN_COMPARATOR_SHA_OR_ROLE_MISMATCH:{name}")
                predictions[name] = values
                if name == "FROZEN_C2_FULL_ROUTER":
                    selected = entry.get("selected_action")
                    if not isinstance(selected, list) or len(selected) != len(outer_features) or not set(selected) <= {"A0", "A1"}:
                        raise ValueError("FROZEN_C2_ACTION_ROLE_OR_KEY_MISMATCH")
                    frozen_action = np.asarray([1 if x == "A1" else 0 for x in selected], dtype=np.int8)
                    if not np.allclose(selected_prediction(a0, a1, frozen_action), values, atol=0, rtol=0):
                        raise ValueError("FROZEN_C2_ACTION_PREDICTION_MISMATCH")
                    actions[name] = frozen_action
                frozen_status[name] = "HASH_VERIFIED"
                frozen_source_provenance[name] = {"source_file_sha": entry["source_file_sha"], "protocol_sha": entry["protocol_sha"], "prediction_sha": entry["sha256"], "router_id": entry.get("router_id"), "selected_action_sha": digest(entry["selected_action"]) if name == "FROZEN_C2_FULL_ROUTER" else None}
            else:
                frozen_status[name] = "NOT_ESTIMABLE_MISSING_FROZEN_SOURCE"
        # Every final strategy is a child of the source meta labels that could
        # have affected it; validation keys are explicit, never outer labels.
        strategy_shas = []
        score_strategy_shas = {}
        for name, selection in selections.items():
            geom = "G0" if "_G0_" in name or name.startswith("G0_MATCHED_") else "G1"
            parent_keys = [t["cache_key"] for t in tasks if (t["purpose"] in ("final_meta", "router_train_meta") and t["geometry"] == geom) or t["purpose"] == "router_validation_pair"]
            parent_shas = sorted(set(task_shas[k] for k in parent_keys))
            source_states = sorted(set(self.features.state))
            val_keys = [t for t in tasks if t["purpose"] == "router_validation_pair"]
            transform_rows = pd.concat([task_rows[t["cache_key"]] for t in tasks if t["purpose"] == "final_meta" and t["geometry"] == geom], ignore_index=True)
            validation_rows = pd.concat([task_rows[t["cache_key"]] for t in val_keys], ignore_index=True)
            strategy_record = {"artifact_id": f"{outer_state}_{outer_year}:{name}", "role": "strategy_fit", "outer_state": outer_state, "outer_year": outer_year,
                "validation_state": None, "validation_year": None, "label_state": outer_state, "label_year": outer_year,
                "allowed_states": source_states, "excluded_states": [outer_state], "max_train_year": outer_year - 1,
                "outcome_role": "historical_source_meta", "transform_fit_keys": selection.get("matched_final_training_keys") or _keys(transform_rows),
                "hyperparameter_selection_keys": _keys(validation_rows),
                "B_history_keys": _keys(self.B.loc[self.B.year.lt(outer_year)]), **self.shas,
                "parent_artifact_shas": parent_shas, "geometry": geom, "B_role": "history_only", "Z_columns": selection["matrix_columns"],
                "selected_config": selection, "nested_transform_provenance": selection["nested_transform_provenance"], "source_file_shas": self.source_file_shas}
            strategy_sha = self.graph.add(strategy_record)
            strategy_shas.append(strategy_sha)
            if name in benefit_scores:
                score_strategy_shas[name] = strategy_sha
        static_parents = sorted(set(task_shas[t["cache_key"]] for t in tasks if t["purpose"] == "final_meta" and t["geometry"] == "G1"))
        for name, selection in (("SOURCE_SELECTED_STATIC", static_selection), ("SOURCE_SELECTED_SOFT_BLEND", blend_selection)):
            source_record = {"artifact_id": f"{outer_state}_{outer_year}:{name}:source_selection", "role": "action", "outer_state": outer_state, "outer_year": outer_year,
                "validation_state": None, "validation_year": None, "label_state": outer_state, "label_year": outer_year,
                "allowed_states": sorted(set(self.features.state)), "excluded_states": [outer_state], "max_train_year": outer_year - 1,
                "outcome_role": "historical_source_meta", "transform_fit_keys": source_meta_keys, "hyperparameter_selection_keys": source_meta_keys,
                "B_history_keys": _keys(self.B.loc[self.B.year.lt(outer_year)]), **self.shas, "parent_artifact_shas": static_parents,
                "geometry": "G1", "B_role": "history_only", "Z_columns": [], "source_selection": selection,
                "source_file_shas": self.source_file_shas}
            strategy_shas.append(self.graph.add(source_record))
        action_record = {"artifact_id": f"{outer_state}_{outer_year}:frozen_actions", "role": "action", "outer_state": outer_state, "outer_year": outer_year,
            "validation_state": None, "validation_year": None, "label_state": outer_state, "label_year": outer_year,
            "allowed_states": sorted(set(self.features.state)), "excluded_states": [outer_state], "max_train_year": outer_year - 1,
            "outcome_role": "none", "transform_fit_keys": [], "hyperparameter_selection_keys": [], "B_history_keys": [], **self.shas,
            "parent_artifact_shas": strategy_shas, "geometry": "G1", "B_role": "history_only", "Z_columns": [], "frozen_source_provenance": frozen_source_provenance}
        action_provenance_sha = self.graph.add(action_record)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        graph_path = out / "full_provenance_graph.json"
        self.graph.write(graph_path)
        graph_sha = file_sha(graph_path)
        manifest = freeze_actions(out / "model_artifacts", outer_features.county_fips.to_numpy(), outer_features.year.to_numpy(), actions, predictions, graph_sha, synthetic=synthetic, provenance_path=graph_path)
        score_manifest = freeze_scores(out / "model_artifacts", outer_features.county_fips.to_numpy(), outer_features.year.to_numpy(), benefit_scores, graph_sha, score_strategy_shas)
        result = {"status": "SYNTHETIC_ACTIONS_FROZEN_NO_EVALUATION" if synthetic else "REAL_ACTIONS_SCORES_FROZEN_NO_EVALUATION", "outer_block": f"{outer_state}_{outer_year}", "nested_prediction_contexts": len(tasks), "source_fit_count": sum(c["fit_models"] for c in costs), "source_cache_hits": sum(c["cache_hit"] for c in costs), "source_fit_cache_hits": sum(c["source_fit_cache_hits"] for c in costs), "factorial_cells": 8, "bridge_count": 3, "comparators": list(comps), "frozen_comparators": frozen_status, "information_ladder_diagnostic": ladder_diagnostic, "matched_size_control": matched_size_control, "selection_records": selections, "source_static_selection": static_selection, "source_blend_selection": blend_selection, "action_manifest": str(manifest), "score_manifest": str(score_manifest), "provenance_graph_sha": graph_sha, "action_provenance_sha": action_provenance_sha, "B_history_only": True, "outer_label_opened": False}
        (out / ("synthetic_full_block_report.json" if synthetic else "model_process_report.json")).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return result


# Historical synthetic scheduler excluded from the scientific release.
