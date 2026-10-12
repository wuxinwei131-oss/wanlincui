"""Source quantiles, source-only representations, registered strategy families."""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .loss import action_from_delta, block_weights, equal_block_metric, loss_pair, pinball, selected_prediction, source_selected_blend, source_selected_static
from .provenance import BAD_FEATURE, digest
from .cache import CanonicalSourceFitCache

Q = (.1, .5, .9)
BASE_CONFIG = {"num_leaves": 15, "learning_rate": .03, "min_child_samples": 20, "feature_fraction": .8, "n_estimators": 300, "random_state": 20260913}
GRID = tuple(itertools.product((7, 15), (.03, .07), (50, 100), (100, 300)))


def fit_source_pair(train: pd.DataFrame, predict: pd.DataFrame, feature_groups: dict[str, list[str]], *, label_year: int, excluded_states: set[str], held_label_state: str | None = None, outcome_column: str = "residual_mg_ha", n_trees: int = 300, threads: int = 1, source_rules: dict | None = None, source_fit_cache: CanonicalSourceFitCache | None = None, cache_identity: dict | None = None):
    """Fit source quantiles, optionally exact C1L latest-year nested selection.

    `source_rules` must be the frozen C1L 00B LightGBM rule payload in future
    authorized runs. None is used only by the compact synthetic guard suite.
    """
    if outcome_column not in train or outcome_column in predict or train.empty or predict.empty:
        raise ValueError("SOURCE_OUTCOME_ROLE_INVALID")
    if train.year.max() >= label_year or set(train.state) & excluded_states:
        raise ValueError("SOURCE_GEO_TEMPORAL_LEAKAGE")
    if held_label_state is not None and (set(predict.state) != {held_label_state} or held_label_state not in excluded_states):
        raise ValueError("G1_LABEL_STATE_ROLE_INVALID")
    if source_fit_cache is not None and (source_rules is None or cache_identity is None or not {"feature_registry_sha", "reference_protocol_sha", "code_sha", "source_file_shas", "preprocessing_protocol_sha"} <= set(cache_identity)):
        raise ValueError("INCOMPLETE_FROZEN_SOURCE_FIT_CACHE_IDENTITY")

    def keys_sha(frame: pd.DataFrame) -> str:
        return digest([{"state": str(row.state), "year": int(row.year), "county_fips": str(row.county_fips)}
                       for row in frame[["state", "year", "county_fips"]].itertuples(index=False)])

    def content_sha(frame: pd.DataFrame, columns: list[str]) -> str:
        normalized = frame[columns].replace([np.inf, -np.inf], np.nan)
        values = normalized.astype(object).where(pd.notna(normalized), None).values.tolist()
        return digest({"columns": columns, "rows": values})

    source_identity_cache: dict[tuple[str, str], dict] = {}

    def predict_quantile(fit_rows: pd.DataFrame, target_rows: pd.DataFrame, cols: list[str], q: float,
                         params: dict, phase: str, trees: tuple[int, ...]) -> tuple[dict[int, np.ndarray], bool]:
        spec = None
        if source_fit_cache is not None:
            identity_key = (system, phase)
            if identity_key not in source_identity_cache:
                source_identity_cache[identity_key] = {
                    "exact_training_row_key_sha": keys_sha(fit_rows),
                    "exact_validation_row_key_sha": keys_sha(target_rows) if phase == "candidate" else digest([]),
                    "prediction_target_row_key_sha": keys_sha(target_rows),
                    "source_data_sha": digest({"source_file_shas": cache_identity["source_file_shas"],
                                               "train_content_sha": content_sha(fit_rows, ["state", "year", "county_fips", outcome_column] + cols),
                                               "target_content_sha": content_sha(target_rows, ["state", "year", "county_fips"] + cols)}),
                }
            spec = {
                **source_identity_cache[identity_key],
                "action": system, "quantile": q,
                "feature_registry_sha": cache_identity["feature_registry_sha"],
                "reference_protocol_sha": cache_identity["reference_protocol_sha"],
                "hyperparameter_tuple": {**params, "random_state": source_rules["lightgbm"]["random_state"], "n_jobs": threads},
                "train_year_max": int(fit_rows.year.max()), "label_year": int(label_year), "excluded_states": sorted(excluded_states),
                "code_sha": cache_identity["code_sha"],
                "preprocessing_protocol_sha": cache_identity["preprocessing_protocol_sha"],
                "phase": phase,
            }
            try:
                payload = json.loads(source_fit_cache.get(spec).decode("utf-8"))
                predictions = {int(tree): np.asarray(payload[str(tree)], float) for tree in trees}
                if any(len(value) != len(target_rows) or not np.isfinite(value).all() for value in predictions.values()):
                    raise ValueError("CANONICAL_FIT_CACHE_PREDICTION_SHAPE_OR_FINITE_FAIL")
                return predictions, False
            except FileNotFoundError:
                pass
        x_fit = fit_rows[cols].replace([np.inf, -np.inf], np.nan)
        x_target = target_rows[cols].replace([np.inf, -np.inf], np.nan)
        model = lgb.LGBMRegressor(objective="quantile", alpha=q, num_leaves=params["num_leaves"],
                                 learning_rate=params["learning_rate"], min_child_samples=params["min_child_samples"],
                                 colsample_bytree=params["feature_fraction"], n_estimators=max(trees),
                                 random_state=source_rules["lightgbm"]["random_state"] if source_rules is not None else BASE_CONFIG["random_state"],
                                 n_jobs=threads, verbosity=-1, force_col_wise=True)
        model.fit(x_fit, fit_rows[outcome_column].to_numpy(float))
        predictions = {int(tree): model.predict(x_target, num_iteration=int(tree)) for tree in trees}
        if source_fit_cache is not None:
            source_fit_cache.put(spec, json.dumps({str(tree): values.tolist() for tree, values in predictions.items()}, separators=(",", ":")).encode("utf-8"))
        return predictions, True

    output, selections, fit_count = {}, {}, 0
    for system in ("A0", "A1"):
        cols = feature_groups[system]
        if any(BAD_FEATURE.search(c) for c in cols):
            raise ValueError("OUTCOME_DERIVED_BASE_FEATURE")
        for col in cols:
            if col not in train or col not in predict:
                raise ValueError(f"MISSING_BASE_FEATURE:{col}")
        x, xp = train[cols].replace([np.inf, -np.inf], np.nan), predict[cols].replace([np.inf, -np.inf], np.nan)
        ys = train[outcome_column].to_numpy(float)
        selected = None
        if source_rules is not None:
            rules = source_rules["lightgbm"]
            last_year = int(train.year.max())
            inner_train = train.loc[train.year.lt(last_year)]
            inner_val = train.loc[train.year.eq(last_year)]
            if inner_train.empty or inner_val.empty or last_year != label_year - 1:
                raise RuntimeError("NOT_ESTIMABLE_SOURCE_INNER_YEAR")
            if max(rules["n_estimators"]) >= 300 and (len(inner_train) <= 100 or len(inner_val) < 20):
                raise RuntimeError("NOT_ESTIMABLE_C1L_FROZEN_INNER_SUPPORT_THRESHOLD")
            y_inner = inner_train[outcome_column].to_numpy(float)
            y_val = inner_val[outcome_column].to_numpy(float)
            best_score = float("inf")
            for leaves, rate, child, fraction in itertools.product(rules["num_leaves"], rules["learning_rate"], rules["min_child_samples"], rules["feature_fraction"]):
                candidate = {"num_leaves": leaves, "learning_rate": rate, "min_child_samples": child, "feature_fraction": fraction}
                qpred_by_trees = {int(tree): [] for tree in rules["n_estimators"]}
                for q in Q:
                    predicted, did_fit = predict_quantile(inner_train, inner_val, cols, q,
                                                          {**candidate, "n_estimators": max(rules["n_estimators"])},
                                                          "candidate", tuple(int(tree) for tree in rules["n_estimators"]))
                    fit_count += int(did_fit)
                    for tree in rules["n_estimators"]:
                        qpred_by_trees[int(tree)].append(predicted[int(tree)])
                for tree in rules["n_estimators"]:
                    score = float(np.mean(pinball(y_val, np.column_stack(qpred_by_trees[int(tree)]))))
                    if score < best_score - 1e-12:
                        best_score = score
                        selected = {**candidate, "n_estimators": int(tree)}
            if selected is None:
                raise RuntimeError("NOT_ESTIMABLE_SOURCE_SELECTION")
            selections[system] = {"config": selected, "inner_validation_year": last_year, "inner_validation_rows": len(inner_val), "inner_PB3": best_score, "grid_size": int(np.prod([len(rules[k]) for k in ("num_leaves", "learning_rate", "min_child_samples", "feature_fraction", "n_estimators")]))}
        preds = []
        for q in Q:
            params = selected or {**BASE_CONFIG, "n_estimators": n_trees}
            predicted, did_fit = predict_quantile(train, predict, cols, q, params, "final_refit", (int(params["n_estimators"]),))
            fit_count += int(did_fit)
            preds.append(predicted[int(params["n_estimators"])])
        output[system] = np.column_stack(preds)
    return (output, selections, fit_count) if source_rules is not None else output


def build_meta_labels(y: np.ndarray, a0: np.ndarray, a1: np.ndarray, state: np.ndarray, year: np.ndarray) -> pd.DataFrame:
    l0, l1, delta = loss_pair(y, a0, a1)
    return pd.DataFrame({"state": state, "year": year, "L0_PB3": l0, "L1_PB3": l1, "delta": delta, "winner": action_from_delta(delta), "block_weight": block_weights(state, year)})


def build_B_history(history: pd.DataFrame, *, cutoff_year: int, excluded_states: set[str]) -> pd.DataFrame:
    """History-only B channel. No residual/prediction error columns permitted."""
    if any(BAD_FEATURE.search(c) for c in history.columns if c not in ("state", "year", "county_fips")):
        raise ValueError("B_RESIDUAL_MIXING")
    if history.year.max() >= cutoff_year or set(history.state) & excluded_states:
        raise ValueError("B_HISTORY_LEAKAGE")
    return history.copy()


def _numeric_frame(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if any(BAD_FEATURE.search(c) and c != "opt_weather_residual_history_only" for c in cols):
        raise ValueError("TARGET_OUTCOME_IN_Z")
    return frame.reindex(columns=cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


@dataclass
class FoldSafeRepresentation:
    representation: str
    ladder: str
    physical_columns: list[str]
    weather_columns: list[str]
    quality_columns: list[str]
    components: int = 3

    def __post_init__(self) -> None:
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler = StandardScaler()
        self.pca = None
        self.feature_names_: list[str] = []
        self.physical_indices_: list[int] = []
        self.other_indices_: list[int] = []

    def _raw(self, frame: pd.DataFrame) -> pd.DataFrame:
        for sys in ("A0", "A1"):
            for q in (10, 50, 90):
                if f"{sys}_q{q}_mg_ha" not in frame:
                    raise ValueError("MISSING_SOURCE_PREDICTION")
        result = pd.DataFrame(index=frame.index)
        for q in (10, 50, 90):
            result[f"A0_q{q}_mg_ha"] = frame[f"A0_q{q}_mg_ha"]
        result["A0_width"] = frame.A0_q90_mg_ha - frame.A0_q10_mg_ha
        for col in self.weather_columns + self.quality_columns:
            result[col] = frame[col] if col in frame else np.nan
        if self.ladder in ("Z1", "Z2", "Z3"):
            for col in self.physical_columns:
                result[col] = frame[col] if col in frame else np.nan
        if self.ladder in ("Z2", "Z3"):
            for q in (10, 50, 90):
                result[f"disagreement_q{q}"] = frame[f"A1_q{q}_mg_ha"] - frame[f"A0_q{q}_mg_ha"]
            result["width_difference"] = (frame.A1_q90_mg_ha - frame.A1_q10_mg_ha) - result.A0_width
            result["A0_crossing"] = ((frame.A0_q10_mg_ha > frame.A0_q50_mg_ha) | (frame.A0_q50_mg_ha > frame.A0_q90_mg_ha)).astype(float)
            result["A1_crossing"] = ((frame.A1_q10_mg_ha > frame.A1_q50_mg_ha) | (frame.A1_q50_mg_ha > frame.A1_q90_mg_ha)).astype(float)
        if self.ladder == "Z3":
            result["opt_weather_residual_history_only"] = frame.get("opt_weather_residual_history_only", pd.Series(np.nan, index=frame.index))
            for a, b in zip(self.physical_columns[:-1], self.physical_columns[1:]):
                result[f"diff_{b}_{a}"] = result[b] - result[a]
        return _numeric_frame(result, list(result.columns))

    def fit(self, train: pd.DataFrame) -> "FoldSafeRepresentation":
        raw = self._raw(train)
        self.feature_names_ = list(raw.columns)
        x = self.scaler.fit_transform(self.imputer.fit_transform(raw))
        self.physical_indices_ = [i for i, name in enumerate(self.feature_names_) if name in self.physical_columns]
        self.other_indices_ = [i for i in range(len(self.feature_names_)) if i not in self.physical_indices_]
        if self.representation == "R0" and self.physical_columns and self.ladder != "Z0":
            self.pca = PCA(n_components=min(self.components, x.shape[0], len(self.physical_indices_)), random_state=20260914).fit(x[:, self.physical_indices_])
        elif self.representation not in ("R0", "R1"):
            raise ValueError("UNKNOWN_REPRESENTATION")
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self._raw(frame).reindex(columns=self.feature_names_)
        x = self.scaler.transform(self.imputer.transform(raw))
        return np.column_stack((x[:, self.other_indices_], self.pca.transform(x[:, self.physical_indices_]))) if self.pca is not None else x


def weather_optical_residual(train: pd.DataFrame, predict: pd.DataFrame, weather_cols: list[str], optical_col: str, *, label_year: int, excluded_states: set[str]) -> np.ndarray:
    if train.year.max() >= label_year or set(train.state) & excluded_states:
        raise ValueError("Z3_PAST_SOURCE_FIT_LEAKAGE")
    if optical_col not in train or optical_col not in predict:
        return np.full(len(predict), np.nan)
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    x = imp.fit_transform(_numeric_frame(train, weather_cols))
    xp = imp.transform(_numeric_frame(predict, weather_cols))
    y = pd.to_numeric(train[optical_col], errors="coerce").to_numpy(float)
    good = np.isfinite(y)
    if good.sum() < 5:
        return np.full(len(predict), np.nan)
    model = Ridge(alpha=1.).fit(x[good], y[good])
    return pd.to_numeric(predict[optical_col], errors="coerce").to_numpy(float) - model.predict(xp)


def fit_router(train_x: np.ndarray, train_delta: np.ndarray, train_weights: np.ndarray, validation_x: np.ndarray, validation_l0: np.ndarray, validation_l1: np.ndarray, validation_state: np.ndarray, validation_year: np.ndarray, *, objective: str, selection_metric: str | None = None, synthetic_tree_limit: int | None = None) -> dict:
    """Select frozen grid by source-only validation; L1 minimizes mean-sign MAE, L2 selected PB3."""
    if objective not in ("L1", "L2"):
        raise ValueError("UNKNOWN_OBJECTIVE")
    if selection_metric is None:
        selection_metric = "MAE" if objective == "L1" else "SELECTED_PB3"
    if selection_metric not in ("MAE", "SELECTED_PB3"):
        raise ValueError("UNKNOWN_SELECTION_METRIC")
    candidates = []
    for order, (leaves, rate, child, trees) in enumerate(GRID):
        n_tree = min(trees, synthetic_tree_limit) if synthetic_tree_limit else trees
        model = lgb.LGBMRegressor(objective="regression_l1" if objective == "L1" else "regression", num_leaves=leaves, learning_rate=rate, min_child_samples=child, colsample_bytree=1., n_estimators=n_tree, random_state=20260913, n_jobs=1, verbosity=-1, force_col_wise=True)
        model.fit(train_x, train_delta, sample_weight=train_weights)
        pred = model.predict(validation_x)
        if selection_metric == "MAE":
            score = equal_block_metric(np.abs(validation_l0 - validation_l1 - pred), validation_state, validation_year)
        else:
            selected = np.where(action_from_delta(pred) == 1, validation_l1, validation_l0)
            score = equal_block_metric(selected, validation_state, validation_year)
        candidates.append((score, order, model, {"num_leaves": leaves, "learning_rate": rate, "min_child_samples": child, "trees": trees}))
    score, _, model, config = min(candidates, key=lambda x: (x[0], x[1]))
    return {"model": model, "config": config, "validation_score": score, "objective": objective, "selection_metric": selection_metric}


def fit_comparators(train_x: np.ndarray, train_delta: np.ndarray, weights: np.ndarray, validation_x: np.ndarray, validation_l0: np.ndarray, validation_l1: np.ndarray, validation_state: np.ndarray, validation_year: np.ndarray) -> dict:
    result: dict = {}
    class_y = action_from_delta(train_delta)
    for alpha in (.1, 1., 10., 100.):
        model = Ridge(alpha=alpha).fit(train_x, train_delta, sample_weight=weights)
        action = action_from_delta(model.predict(validation_x))
        score = equal_block_metric(np.where(action == 1, validation_l1, validation_l0), validation_state, validation_year)
        result[("ridge", alpha)] = (score, model)
    if len(np.unique(class_y)) == 1:
        action = np.full(len(validation_x), int(class_y[0]))
        score = equal_block_metric(np.where(action == 1, validation_l1, validation_l0), validation_state, validation_year)
        result[("weighted_logit", 0.)] = (score, int(class_y[0]))
    else:
        for c in (.1, 1., 10.):
            model = LogisticRegression(C=c, random_state=20260913, max_iter=1000).fit(train_x, class_y, sample_weight=weights * np.abs(train_delta))
            action = model.predict(validation_x)
            score = equal_block_metric(np.where(action == 1, validation_l1, validation_l0), validation_state, validation_year)
            result[("weighted_logit", c)] = (score, model)
    tree = DecisionTreeClassifier(max_depth=2, random_state=20260913).fit(train_x, class_y, sample_weight=weights * np.abs(train_delta))
    action = tree.predict(validation_x)
    result[("depth2_tree", 2)] = (equal_block_metric(np.where(action == 1, validation_l1, validation_l0), validation_state, validation_year), tree)
    return {family: min(((score, arg, model) for (name, arg), (score, model) in result.items() if name == family), key=lambda x: (np.isnan(x[0]), x[0], x[1])) for family in ("ridge", "weighted_logit", "depth2_tree")}


def bridge_scores(meta: pd.DataFrame, predictions: dict[str, np.ndarray]) -> dict:
    """B1/2/3 bridge reporting; all scores are source-only validation values."""
    s, y = meta.state.to_numpy(), meta.year.to_numpy()
    result = {}
    for name in ("B1_L1_equalblock_MAE", "B2_L1_equalblock_selected_PB3", "B3_L2_equalblock_selected_PB3"):
        pred = predictions[name]
        if name.startswith("B1"):
            score = equal_block_metric(np.abs(meta.delta.to_numpy() - pred), s, y)
        else:
            action = action_from_delta(pred)
            score = equal_block_metric(np.where(action == 1, meta.L1_PB3, meta.L0_PB3), s, y)
        result[name] = score
    return result
