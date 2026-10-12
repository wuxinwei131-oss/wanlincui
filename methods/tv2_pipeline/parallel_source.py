"""Exact C1L source grid in four independent, single-thread LightGBM workers.

Only prediction arrays cross the process boundary; cache entries contain no
model dumps.  The parent makes the frozen 300/700 selection in grid order.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .cache import CanonicalSourceFitCache
from .loss import pinball
from .provenance import BAD_FEATURE, digest

Q = (.1, .5, .9)


def _keys_sha(frame):
    return digest([{"state": str(r.state), "year": int(r.year), "county_fips": str(r.county_fips)}
                   for r in frame[["state", "year", "county_fips"]].itertuples(index=False)])


def _content_sha(frame, cols):
    normalized = frame[cols].replace([np.inf, -np.inf], np.nan)
    return digest({"columns": cols, "rows": normalized.astype(object).where(pd.notna(normalized), None).values.tolist()})


def _fit_batch(job):
    system, q, phase, fit_rows, target_rows, cols, params_list, trees, label_year, excluded, rules, cache_identity, cache_dir, outcome_column = job
    cache = CanonicalSourceFitCache(cache_dir)
    x_fit = fit_rows[cols].replace([np.inf, -np.inf], np.nan)
    x_target = target_rows[cols].replace([np.inf, -np.inf], np.nan)
    source_data_sha = digest({"source_file_shas": cache_identity["source_file_shas"],
                             "train_content_sha": _content_sha(fit_rows, ["state", "year", "county_fips", outcome_column] + cols),
                             "target_content_sha": _content_sha(target_rows, ["state", "year", "county_fips"] + cols)})
    common = {"exact_training_row_key_sha": _keys_sha(fit_rows),
              "exact_validation_row_key_sha": _keys_sha(target_rows) if phase == "candidate" else digest([]),
              "prediction_target_row_key_sha": _keys_sha(target_rows),
              "source_data_sha": source_data_sha, "action": system, "quantile": q,
              "feature_registry_sha": cache_identity["feature_registry_sha"],
              "reference_protocol_sha": cache_identity["reference_protocol_sha"],
              "train_year_max": int(fit_rows.year.max()), "label_year": int(label_year),
              "excluded_states": sorted(excluded), "code_sha": cache_identity["code_sha"],
              "preprocessing_protocol_sha": cache_identity["preprocessing_protocol_sha"], "phase": phase}
    out, fit_count = [], 0
    for params in params_list:
        fit_trees = (int(params["n_estimators"]),) if phase == "final_refit" else trees
        spec = {**common, "hyperparameter_tuple": {**params, "random_state": rules["random_state"], "n_jobs": 1}}
        try:
            payload = json.loads(cache.get(spec).decode("utf-8"))
            predictions = {int(tree): np.asarray(payload[str(tree)], float) for tree in fit_trees}
            if any(len(a) != len(target_rows) or not np.isfinite(a).all() for a in predictions.values()):
                raise ValueError("CANONICAL_FIT_CACHE_PREDICTION_SHAPE_OR_FINITE_FAIL")
        except FileNotFoundError:
            model = lgb.LGBMRegressor(objective="quantile", alpha=q, num_leaves=params["num_leaves"],
                                     learning_rate=params["learning_rate"], min_child_samples=params["min_child_samples"],
                                     colsample_bytree=params["feature_fraction"], n_estimators=max(fit_trees),
                                     random_state=rules["random_state"], n_jobs=1, verbosity=-1, force_col_wise=True)
            model.fit(x_fit, fit_rows[outcome_column].to_numpy(float))
            predictions = {int(tree): model.predict(x_target, num_iteration=int(tree)) for tree in fit_trees}
            cache.put(spec, json.dumps({str(tree): values.tolist() for tree, values in predictions.items()}, separators=(",", ":")).encode("utf-8"))
            fit_count += 1
        out.append(predictions)
    return system, q, out, fit_count


def fit_source_pair_parallel(train, predict, feature_groups, *, label_year, excluded_states,
                             held_label_state, source_rules, source_fit_cache, cache_identity,
                             executor, outcome_column="residual_mg_ha"):
    if source_rules is None or source_fit_cache is None or cache_identity is None:
        raise ValueError("PARALLEL_FROZEN_SOURCE_RULES_CACHE_REQUIRED")
    if outcome_column not in train or outcome_column in predict or train.empty or predict.empty:
        raise ValueError("SOURCE_OUTCOME_ROLE_INVALID")
    if train.year.max() >= label_year or set(train.state) & excluded_states:
        raise ValueError("SOURCE_GEO_TEMPORAL_LEAKAGE")
    if held_label_state is not None and (set(predict.state) != {held_label_state} or held_label_state not in excluded_states):
        raise ValueError("G1_LABEL_STATE_ROLE_INVALID")
    if not {"feature_registry_sha", "reference_protocol_sha", "code_sha", "source_file_shas", "preprocessing_protocol_sha"} <= set(cache_identity):
        raise ValueError("INCOMPLETE_FROZEN_SOURCE_FIT_CACHE_IDENTITY")
    for system in ("A0", "A1"):
        if any(BAD_FEATURE.search(c) for c in feature_groups[system]):
            raise ValueError("OUTCOME_DERIVED_BASE_FEATURE")
        for col in feature_groups[system]:
            if col not in train or col not in predict:
                raise ValueError(f"MISSING_BASE_FEATURE:{col}")
    rules = source_rules["lightgbm"]
    last_year = int(train.year.max())
    inner_train = train.loc[train.year.lt(last_year)]
    inner_val = train.loc[train.year.eq(last_year)]
    if inner_train.empty or inner_val.empty or last_year != label_year - 1:
        raise RuntimeError("NOT_ESTIMABLE_SOURCE_INNER_YEAR")
    if max(rules["n_estimators"]) >= 300 and (len(inner_train) <= 100 or len(inner_val) < 20):
        raise RuntimeError("NOT_ESTIMABLE_C1L_FROZEN_INNER_SUPPORT_THRESHOLD")
    candidates = [{"num_leaves": a, "learning_rate": b, "min_child_samples": c,
                   "feature_fraction": d, "n_estimators": max(rules["n_estimators"])}
                  for a, b, c, d in itertools.product(rules["num_leaves"], rules["learning_rate"],
                                                       rules["min_child_samples"], rules["feature_fraction"])]
    def jobs(phase, fit, target, param_by_system, trees):
        # Four-config chunks distribute 96 candidate fits evenly across four
        # processes while retaining grid order for deterministic parent scoring.
        return [(system, q, phase, fit, target, feature_groups[system], params[start:start+4], trees,
                 label_year, excluded_states, rules, cache_identity, str(source_fit_cache.root), outcome_column)
                for system in ("A0", "A1") for q in Q
                for params in (param_by_system[system],)
                for start in range(0, len(params), 4)]
    results = list(executor.map(_fit_batch, jobs("candidate", inner_train, inner_val,
                                                {"A0": candidates, "A1": candidates},
                                                tuple(int(x) for x in rules["n_estimators"]))))
    by = {}
    for system, q, values, _ in results:
        by.setdefault((system, q), []).extend(values)
    fit_count = sum(n for _, _, _, n in results)
    selected, selections = {}, {}
    y_val = inner_val[outcome_column].to_numpy(float)
    for system in ("A0", "A1"):
        best = float("inf")
        for idx, candidate in enumerate(candidates):
            for tree in rules["n_estimators"]:
                pred = np.column_stack([by[(system, q)][idx][int(tree)] for q in Q])
                score = float(np.mean(pinball(y_val, pred)))
                if score < best - 1e-12:
                    best = score
                    selected[system] = {**candidate, "n_estimators": int(tree)}
        selections[system] = {"config": selected[system], "inner_validation_year": last_year,
                              "inner_validation_rows": len(inner_val), "inner_PB3": best,
                              "grid_size": int(np.prod([len(rules[k]) for k in ("num_leaves", "learning_rate", "min_child_samples", "feature_fraction", "n_estimators")]))}
    finals = list(executor.map(_fit_batch, jobs("final_refit", train, predict,
                                              {s: [selected[s]] for s in ("A0", "A1")},
                                              (int(selected["A0"]["n_estimators"]), int(selected["A1"]["n_estimators"])))) )
    fit_count += sum(n for _, _, _, n in finals)
    output = {system: np.column_stack([next(values[0][selected[system]["n_estimators"]]
                                       for s, q2, values, _ in finals if s == system and q2 == q)
                                       for q in Q]) for system in ("A0", "A1")}
    return output, selections, fit_count
