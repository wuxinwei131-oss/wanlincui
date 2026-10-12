"""QA1-corrected fixed L2/G0/R1 soybean benefit router; Z1/Z2 only."""
from __future__ import annotations
from _paths import method_path, method_config, earth_engine_project

import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from tv5_source_model import Inputs, ROOT
from tv2_pipeline.loss import action_from_delta, block_weights, equal_block_metric
from tv2_pipeline.models import FoldSafeRepresentation, GRID
from tv2_pipeline.provenance import digest

Q = (10, 50, 90)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def concat_meta(inputs: Inputs, states: tuple[str, ...], years: range) -> pd.DataFrame:
    parts = [inputs.meta(allowed=states, target_state=state, target_year=year, geometry="G0")
             for year in years for state in states]
    frame = pd.concat(parts, ignore_index=True)
    # QA1 correction: weights are recomputed after every concat and any label loss.
    frame["block_weight"] = block_weights(frame.state.to_numpy(), frame.year.to_numpy())
    masses = frame.groupby(["state", "year"]).block_weight.sum()
    if not np.isclose(float(frame.block_weight.mean()), 1.0, atol=1e-10) or not np.allclose(masses, masses.iloc[0], atol=1e-9):
        raise RuntimeError("Corrected equal-block learner weights failed")
    return frame


def condition_domain(inputs: Inputs, condition: str, outer_state: str) -> tuple[tuple[str, ...], str | None]:
    states = inputs.states
    if condition == "SOURCE_DOMAIN_TEMPORAL":
        if outer_state not in states:
            raise RuntimeError("Source-temporal target not a development state")
        partner = states[(states.index(outer_state) + 1) % len(states)]
        return tuple(s for s in states if s != partner), partner
    if condition == "UNSEEN_GEOGRAPHY":
        if outer_state not in states:
            raise RuntimeError("Unseen target not a development state")
        return tuple(s for s in states if s != outer_state), None
    if condition == "EXTERNAL_CONFIRMATION":
        if outer_state != "EXTERNAL_ALL":
            raise RuntimeError("External target identity invalid")
        return states, None
    raise RuntimeError("Unregistered condition")


def fit_one(inputs: Inputs, *, condition: str, outer_state: str, z: str) -> tuple[pd.DataFrame, dict]:
    if z not in ("Z1", "Z2") or (condition != "UNSEEN_GEOGRAPHY" and z != "Z2"):
        raise RuntimeError("Unregistered TV5 information condition")
    year = inputs.year
    domain, partner = condition_domain(inputs, condition, outer_state)
    if condition == "SOURCE_DOMAIN_TEMPORAL":
        outer = inputs.prediction(allowed=domain, target_state=outer_state, target_year=year, geometry="G0")
    elif condition == "UNSEEN_GEOGRAPHY":
        outer = inputs.prediction(allowed=domain, target_state=outer_state, target_year=year, geometry="G1")
    else:
        outer = inputs.prediction(allowed=domain, target_state="EXTERNAL_ALL", target_year=year,
                                  geometry="G1", external=True, include_as=True)
    final = concat_meta(inputs, domain, range(2016, year))
    weather = list(dict.fromkeys(inputs.groups["A0"] + ["prior_soybean_effective_acres"]))
    original_lock = json.loads((Path(str(method_path('reconstruction'))) / "TV1_RESEARCH_DESIGN_LOCK.json").read_text(encoding="utf-8"))
    physical = original_lock["physical_columns"]
    quality = original_lock["quality_columns"]
    folds = []
    for validation_year in range(year - 2, year):
        for validation_state in domain:
            inner = tuple(s for s in domain if s != validation_state)
            tr = concat_meta(inputs, inner, range(2016, validation_year))
            va = inputs.meta(allowed=inner, target_state=validation_state,
                             target_year=validation_year, geometry="G1")
            if tr.empty or va.empty or tr.year.max() >= validation_year:
                raise RuntimeError("Nested validation chronology/support failure")
            transformer = FoldSafeRepresentation("R1", z, physical, weather, quality).fit(tr)
            folds.append({"validation_state": validation_state, "validation_year": validation_year,
                          "train_x": transformer.transform(tr), "train_y": tr.delta.to_numpy(float),
                          "train_weight": tr.block_weight.to_numpy(float),
                          "validation_x": transformer.transform(va),
                          "validation_L0": va.L0_PB3.to_numpy(float),
                          "validation_L1": va.L1_PB3.to_numpy(float),
                          "train_rows": len(tr), "validation_rows": len(va)})
    if not folds:
        raise RuntimeError("No registered upper validation folds")
    final_transformer = FoldSafeRepresentation("R1", z, physical, weather, quality).fit(final)
    final_x, outer_x = final_transformer.transform(final), final_transformer.transform(outer)
    grid_scores = []
    fits = 0
    for order, (leaves, rate, child, trees) in enumerate(GRID):
        validation_scores = []
        for fold in folds:
            model = lgb.LGBMRegressor(objective="regression", num_leaves=leaves,
                                      learning_rate=rate, min_child_samples=child,
                                      colsample_bytree=1.0, n_estimators=trees,
                                      random_state=20260913, n_jobs=1,
                                      verbosity=-1, force_col_wise=True)
            model.fit(fold["train_x"], fold["train_y"], sample_weight=fold["train_weight"])
            fits += 1
            score = model.predict(fold["validation_x"])
            selected = np.where(action_from_delta(score) == 1, fold["validation_L1"], fold["validation_L0"])
            validation_scores.append(float(np.mean(selected)))
        grid_scores.append((float(np.mean(validation_scores)), order, (leaves, rate, child, trees)))
    selected_score, _, (leaves, rate, child, trees) = min(grid_scores)
    model = lgb.LGBMRegressor(objective="regression", num_leaves=leaves,
                              learning_rate=rate, min_child_samples=child,
                              colsample_bytree=1.0, n_estimators=trees,
                              random_state=20260913, n_jobs=1,
                              verbosity=-1, force_col_wise=True)
    model.fit(final_x, final.delta.to_numpy(float), sample_weight=final.block_weight.to_numpy(float))
    fits += 1
    score = model.predict(outer_x)
    actions = action_from_delta(score)
    keep = ["state", "year", "county_fips", "B_history_mg_ha", "max_history_year"] + [f"{a}_q{q}_mg_ha" for a in (("A0", "A1", "AS") if condition == "EXTERNAL_CONFIRMATION" else ("A0", "A1")) for q in Q]
    result = outer[keep].copy()
    result["benefit_score"] = score
    result["action"] = np.where(actions == 1, "A1", "A0")
    constant = equal_block_metric(final.delta.to_numpy(float), final.state.to_numpy(), final.year.to_numpy())
    result["source_constant_score"] = constant
    result["source_static_action"] = "A1" if constant > 0 else "A0"
    result["condition"] = condition
    result["Z"] = z
    if result.duplicated(["state", "year", "county_fips"]).any() or not result.year.eq(year).all():
        raise RuntimeError("Frozen upper prediction keys invalid")
    masses = final.groupby(["state", "year"]).block_weight.sum()
    report = {"condition": condition, "outer_state": outer_state, "outer_year": year, "Z": z,
              "source_excluded_partner": partner, "outer_rows": len(result),
              "final_meta_rows": len(final), "final_meta_blocks": len(masses),
              "corrected_final_block_weight_mean": float(final.block_weight.mean()),
              "corrected_final_block_mass_range": [float(masses.min()), float(masses.max())],
              "validation_folds": len(folds),
              "fold_support": [{k: v for k, v in fold.items() if k in ("validation_state", "validation_year", "train_rows", "validation_rows")} for fold in folds],
              "grid_size": len(GRID), "upper_fit_count": fits,
              "selected_grid": [leaves, rate, child, trees],
              "source_validation_selected_PB3": selected_score,
              "matrix_feature_count": int(final_x.shape[1]),
              "exact_registry_columns": list(final_transformer.feature_names_),
              "exact_registry_sha256": digest(list(final_transformer.feature_names_)),
              "QA1_corrected_models_sha256": sha(Path(str(method_path('qa1_code') / 'tv2_pipeline/models.py'))),
              "TV5_upper_code_sha256": sha(Path(__file__)),
              "current_target_outcome_accessed": False}
    return result, report
