"""Fail-closed provenance graph for source fits, meta labels and actions."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

STATES = frozenset(("IA", "IL", "IN", "KS", "NE", "TX"))
OUTER_STATES = STATES | {"EXTERNAL"}
SOURCE_ROLES = {"source_fit", "reference_history", "feature", "transform_fit", "hyperparameter_selection", "meta_label", "strategy_fit", "action"}
SHA_FIELDS = ("feature_registry_sha", "reference_protocol_sha", "base_config_sha", "fold_selection_sha", "code_sha")
REQUIRED = ("artifact_id", "role", "outer_state", "outer_year", "validation_state", "validation_year", "label_state", "label_year", "allowed_states", "excluded_states", "max_train_year", "outcome_role", "transform_fit_keys", "hyperparameter_selection_keys", "B_history_keys", *SHA_FIELDS, "parent_artifact_shas")
BAD_FEATURE = re.compile(r"(^|_)(yield|target|residual|benefit|winner|pb3|loss|observed|outcome)(_|$)", re.I)


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_tree_sha(root: str | Path | None = None) -> str:
    """Bind the CLI, all pipeline modules and disabled adapter to one code SHA."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    files = sorted(root.glob("*.py")) + sorted((root / "tv2_pipeline").glob("*.py"))
    if not files or not any(p.name == "tv2_runner.py" for p in files):
        raise ValueError("RUNNER_CODE_TREE_INCOMPLETE")
    return digest({p.relative_to(root).as_posix(): file_sha(p) for p in files})


def safe_path(path: str | Path, *, outcome: bool = False) -> Path:
    p = Path(path).resolve(strict=True)
    s = str(p).lower().replace("\\", "/")
    if outcome and (re.search(r"(?:^|[/_\-.])(2025|2026)(?:[/_\-.]|$)", s) or "p0_" in s or "prospective" in s):
        raise ValueError("FORBIDDEN_PROJECT_2025_2026_OR_P0_OUTCOME_PATH")
    return p


def _validate_record(rec: dict) -> None:
    missing = [x for x in REQUIRED if x not in rec]
    if missing:
        raise ValueError(f"INCOMPLETE_PROVENANCE:{missing}")
    if rec["role"] not in SOURCE_ROLES or not all(rec.get(k) for k in SHA_FIELDS):
        raise ValueError("INVALID_ROLE_OR_SOURCE_SHA")
    if rec["role"] in {"source_fit", "meta_label"} and (int(rec.get("effective_tree_count", 0)) <= 0 or rec.get("execution_mode") not in {"SYNTHETIC_ONLY", "FROZEN_300_TREES", "FROZEN_C1L_300_700_GRID"} or int(rec.get("threads", 0)) <= 0):
        raise ValueError("SOURCE_FIT_EFFECTIVE_CONFIG_PROVENANCE_MISSING")
    if rec["role"] in {"source_fit", "meta_label"} and (not rec.get("source_file_shas") or len(rec.get("source_data_sha", "")) != 64 or any(len(str(value)) != 64 for value in rec["source_file_shas"].values())):
        raise ValueError("SOURCE_FIT_DATA_SHA_PROVENANCE_MISSING")
    if rec["outer_state"] not in OUTER_STATES or rec["label_state"] not in OUTER_STATES:
        raise ValueError("UNKNOWN_STATE")
    if rec["validation_state"] is not None and rec["validation_state"] not in STATES:
        raise ValueError("UNKNOWN_VALIDATION_STATE")
    if int(rec["outer_year"]) > 2024 or int(rec["label_year"]) > 2024:
        raise ValueError("FORBIDDEN_2025_2026_YEAR")
    if rec["validation_year"] is not None and int(rec["validation_year"]) >= int(rec["outer_year"]):
        raise ValueError("VALIDATION_NOT_BEFORE_OUTER")
    allowed, excluded = set(rec["allowed_states"]), set(rec["excluded_states"])
    if not allowed <= STATES or not excluded <= OUTER_STATES or allowed & excluded:
        raise ValueError("HELD_STATE_IN_ALLOWED_SET")
    if rec["outer_state"] not in excluded or rec["outer_state"] in allowed:
        raise ValueError("OUTER_STATE_ANCESTOR_LEAKAGE")
    if rec["validation_state"] is not None and (rec["validation_state"] not in excluded or rec["validation_state"] in allowed):
        raise ValueError("INNER_VALIDATION_STATE_ANCESTOR_LEAKAGE")
    if rec.get("geometry") == "G1" and rec["label_state"] not in excluded:
        raise ValueError("G1_LABEL_STATE_NOT_EXCLUDED")
    if int(rec["max_train_year"]) >= int(rec["label_year"]):
        raise ValueError("FUTURE_OR_SAME_YEAR_TRAINING")
    if any(BAD_FEATURE.search(str(c)) and str(c) != "opt_weather_residual_history_only" for c in rec.get("Z_columns", [])):
        raise ValueError("TARGET_OUTCOME_IN_Z")
    if rec.get("B_role") not in (None, "history_only") or rec.get("residual_feature_role") == "B":
        raise ValueError("B_RESIDUAL_MIXING")
    if rec["outcome_role"] in {"outer_evaluation", "prospective", "held_label"} and rec["role"] != "action":
        raise ValueError("EVALUATION_OUTCOME_IN_MODEL_GRAPH")
    for key in ("transform_fit_keys", "hyperparameter_selection_keys", "B_history_keys"):
        for item in rec[key]:
            if item.get("state") in excluded or int(item.get("year", 9999)) >= int(rec["label_year"]):
                raise ValueError(f"LEAKAGE_IN_{key}")
    if rec["role"] == "strategy_fit":
        folds = rec.get("nested_transform_provenance")
        if not isinstance(folds, dict) or not folds:
            raise ValueError("NESTED_TRANSFORM_PROVENANCE_MISSING")
        for fold_id, fold in folds.items():
            required = ("validation_keys_sha", "validation_keys", "transform_train_keys_sha",
                        "transform_train_keys", "transform_train_states", "transform_train_year_max",
                        "representation", "transform_object_sha", "parent_source_artifact_sha",
                        "parent_source_artifact_shas")
            if any(k not in fold for k in required):
                raise ValueError("NESTED_TRANSFORM_FIELD_MISSING")
            validation_keys, train_keys = fold["validation_keys"], fold["transform_train_keys"]
            if not validation_keys or not train_keys or digest(validation_keys) != fold["validation_keys_sha"] or digest(train_keys) != fold["transform_train_keys_sha"]:
                raise ValueError("NESTED_TRANSFORM_KEY_SHA_MISMATCH")
            validation_pairs = {(str(k["state"]), int(k["year"])) for k in validation_keys}
            if len(validation_pairs) != 1:
                raise ValueError("NESTED_VALIDATION_PAIR_AMBIGUOUS")
            validation_state, validation_year = next(iter(validation_pairs))
            if fold_id != f"{validation_state}_{validation_year}" or validation_state == rec["outer_state"]:
                raise ValueError("NESTED_VALIDATION_FOLD_ID_INVALID")
            if any(str(k["state"]) in {rec["outer_state"], validation_state} or int(k["year"]) >= validation_year for k in train_keys):
                raise ValueError("NESTED_TRANSFORM_HELD_OR_FUTURE_LEAKAGE")
            if int(fold["transform_train_year_max"]) != max(int(k["year"]) for k in train_keys) or fold["transform_train_year_max"] >= validation_year:
                raise ValueError("NESTED_TRANSFORM_YEAR_MAX_MISMATCH")
            if sorted(set(str(k["state"]) for k in train_keys)) != fold["transform_train_states"]:
                raise ValueError("NESTED_TRANSFORM_STATES_MISMATCH")
            if fold["representation"] != rec["selected_config"].get("representation") or len(fold["transform_object_sha"]) != 64:
                raise ValueError("NESTED_TRANSFORM_OBJECT_OR_REPRESENTATION_INVALID")
            source_parents = sorted(fold["parent_source_artifact_shas"])
            if not source_parents or digest(source_parents) != fold["parent_source_artifact_sha"] or not set(source_parents) <= set(rec["parent_artifact_shas"]):
                raise ValueError("NESTED_TRANSFORM_PARENT_SHA_INVALID")
    for src in rec.get("source_files", []):
        p = safe_path(src["path"], outcome=src.get("outcome", False))
        if file_sha(p) != src["sha256"]:
            raise ValueError("SOURCE_SHA_MISMATCH")


class ProvenanceGraph:
    def __init__(self) -> None:
        self.records: dict[str, dict] = {}

    def add(self, rec: dict) -> str:
        _validate_record(rec)
        parents = rec["parent_artifact_shas"]
        if any(p not in self.records for p in parents):
            raise ValueError("PARENT_SHA_MISSING")
        for p in parents:
            parent = self.records[p]
            if set(parent["allowed_states"]) & set(rec["excluded_states"]):
                raise ValueError("EXCLUDED_STATE_IN_ANCESTOR")
            if int(parent["max_train_year"]) >= int(rec["label_year"]):
                raise ValueError("FUTURE_YEAR_IN_ANCESTOR")
            if parent["outcome_role"] in {"outer_evaluation", "prospective", "held_label"}:
                raise ValueError("EVALUATION_LABEL_IN_ANCESTOR")
        identity = digest(rec)
        self.records[identity] = rec
        return identity

    def verify(self) -> None:
        original = self.records
        replay = ProvenanceGraph()
        for key, rec in original.items():
            if replay.add(rec) != key:
                raise ValueError("PROVENANCE_GRAPH_SHA_MISMATCH")

    def descendants(self, parent_sha: str) -> set[str]:
        found = {parent_sha}
        changed = True
        while changed:
            changed = False
            for k, rec in self.records.items():
                if k not in found and found.intersection(rec["parent_artifact_shas"]):
                    found.add(k); changed = True
        return found

    def write(self, path: str | Path) -> None:
        self.verify()
        Path(path).write_text(json.dumps({"schema": "TV1R2.provenance.v1", "records": self.records}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
