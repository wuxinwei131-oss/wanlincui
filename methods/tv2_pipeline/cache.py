"""Semantic cache; byte and ancestry hashes are checked on every read."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

from .provenance import digest, file_sha

KEY_FIELDS = ("allowed_states", "excluded_states", "outer_state", "outer_year", "validation_state", "validation_year", "label_state", "label_year", "predict_state", "predict_year", "geometry", "representation", "feature_registry_sha", "reference_protocol_sha", "base_config_sha", "fold_selection_sha", "code_sha", "effective_tree_count", "execution_mode", "threads", "parent_artifact_shas")


def semantic_key(task: dict) -> str:
    missing = [k for k in KEY_FIELDS if k not in task]
    if missing or any(not task[k] for k in ("feature_registry_sha", "reference_protocol_sha", "base_config_sha", "fold_selection_sha", "code_sha")):
        raise ValueError(f"INCOMPLETE_SEMANTIC_CACHE_KEY:{missing}")
    return digest({k: task[k] for k in KEY_FIELDS})


class Cache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, task: dict, payload: bytes) -> str:
        key = semantic_key(task)
        data = self.root / f"{key}.bin"
        meta = self.root / f"{key}.json"
        data.write_bytes(payload)
        meta.write_text(json.dumps({"key": key, "task": task, "payload_sha": file_sha(data), "parents": task["parent_artifact_shas"]}, sort_keys=True) + "\n", encoding="utf-8")
        return key

    def get(self, task: dict, *, known_parent_shas: set[str]) -> bytes:
        key = semantic_key(task)
        data = self.root / f"{key}.bin"
        meta = self.root / f"{key}.json"
        found = json.loads(meta.read_text(encoding="utf-8"))
        if found["key"] != key or digest(found["task"]) != digest(task) or found["payload_sha"] != file_sha(data):
            raise ValueError("CACHE_SHA_MISMATCH")
        if not set(found["parents"]) <= known_parent_shas:
            raise ValueError("CACHE_PARENT_SHA_MISMATCH")
        return data.read_bytes()


# A source fit has a content identity independent of the outer task that
# requests it. The old task-graph key above remains valid for graph nodes and
# provenance, but must not be used as a cross-block source-fit cache key.
CANONICAL_FIT_FIELDS = (
    "exact_training_row_key_sha", "exact_validation_row_key_sha",
    "prediction_target_row_key_sha", "action", "quantile",
    "feature_registry_sha", "reference_protocol_sha", "hyperparameter_tuple",
    "train_year_max", "label_year", "excluded_states", "source_data_sha", "code_sha",
    "preprocessing_protocol_sha", "phase",
)


def _fit_spec(spec: dict) -> dict:
    return {name: sorted(spec[name]) if name == "excluded_states" else spec[name]
            for name in CANONICAL_FIT_FIELDS}


def canonical_source_fit_key(spec: dict) -> str:
    missing = [name for name in CANONICAL_FIT_FIELDS if name not in spec]
    if missing:
        raise ValueError(f"INCOMPLETE_CANONICAL_SOURCE_FIT_KEY:{missing}")
    if spec["action"] not in ("A0", "A1") or spec["quantile"] not in (0.1, 0.5, 0.9):
        raise ValueError("UNREGISTERED_ACTION_OR_QUANTILE")
    if spec["phase"] not in ("candidate", "final_refit"):
        raise ValueError("UNREGISTERED_SOURCE_FIT_PHASE")
    if int(spec["train_year_max"]) >= int(spec["label_year"]):
        raise ValueError("SOURCE_FIT_FUTURE_YEAR")
    if len(set(spec["excluded_states"])) != len(spec["excluded_states"]):
        raise ValueError("DUPLICATE_EXCLUDED_STATE")
    return digest(_fit_spec(spec))


class CanonicalSourceFitCache:
    """Prediction-only cache with byte and scientific-parent verification."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, spec: dict, payload: bytes) -> str:
        key = canonical_source_fit_key(spec)
        data = self.root / f"{key}.bin"
        meta = self.root / f"{key}.json"
        data.write_bytes(payload)
        meta.write_text(json.dumps({"key": key, "scientific_spec": _fit_spec(spec),
                                    "payload_sha": hashlib.sha256(payload).hexdigest(),
                                    "parent_source_sha": spec["source_data_sha"]}, sort_keys=True) + "\n", encoding="utf-8")
        return key

    def get(self, spec: dict) -> bytes:
        key = canonical_source_fit_key(spec)
        data = self.root / f"{key}.bin"
        meta = self.root / f"{key}.json"
        found = json.loads(meta.read_text(encoding="utf-8"))
        payload = data.read_bytes()
        if found["key"] != key or found["parent_source_sha"] != spec["source_data_sha"]:
            raise ValueError("CANONICAL_CACHE_PARENT_SHA_MISMATCH")
        if found["payload_sha"] != hashlib.sha256(payload).hexdigest():
            raise ValueError("CANONICAL_CACHE_PAYLOAD_SHA_MISMATCH")
        if digest(found["scientific_spec"]) != digest(_fit_spec(spec)):
            raise ValueError("CANONICAL_CACHE_SPEC_MISMATCH")
        return payload


CONTEXT_FIELDS = (
    "exact_training_row_key_sha", "exact_validation_row_key_sha",
    "prediction_target_row_key_sha", "training_values_and_keys_sha", "target_predictor_and_B_sha",
    "source_files_sha", "allowed_states", "excluded_states", "label_state",
    "label_year", "geometry", "actions", "quantiles", "feature_registry_sha",
    "reference_protocol_sha", "base_config_sha", "source_grid_sha", "code_sha",
    "preprocessing_protocol_sha", "effective_tree_count", "threads", "execution_mode",
)


def _context_spec(spec: dict) -> dict:
    return {name: sorted(spec[name]) if name in ("allowed_states", "excluded_states") else spec[name]
            for name in CONTEXT_FIELDS}


def canonical_source_context_key(spec: dict) -> str:
    missing = [name for name in CONTEXT_FIELDS if name not in spec]
    if missing:
        raise ValueError(f"INCOMPLETE_CANONICAL_CONTEXT:{missing}")
    if spec["actions"] != ["A0", "A1"] or spec["quantiles"] != [0.1, 0.5, 0.9]:
        raise ValueError("CANONICAL_CONTEXT_ACTION_OR_QUANTILE_DRIFT")
    if len(set(spec["allowed_states"])) != len(spec["allowed_states"]) or len(set(spec["excluded_states"])) != len(spec["excluded_states"]) or set(spec["allowed_states"]) & set(spec["excluded_states"]):
        raise ValueError("CANONICAL_CONTEXT_STATE_SET_INVALID")
    return digest(_context_spec(spec))


class CanonicalSourceContextCache:
    """Wrapper-independent A0/A1 context cache with byte/parent verification."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, spec: dict, payload: bytes) -> str:
        key = canonical_source_context_key(spec)
        data = self.root / f"{key}.bin"
        meta = self.root / f"{key}.json"
        data.write_bytes(payload)
        meta.write_text(json.dumps({"key": key, "scientific_spec": _context_spec(spec),
                                    "payload_sha": hashlib.sha256(payload).hexdigest(),
                                    "parent_values_sha": spec["training_values_and_keys_sha"],
                                    "parent_target_sha": spec["target_predictor_and_B_sha"]}, sort_keys=True) + "\n", encoding="utf-8")
        return key

    def get(self, spec: dict) -> bytes:
        key = canonical_source_context_key(spec)
        data = self.root / f"{key}.bin"
        meta = self.root / f"{key}.json"
        found = json.loads(meta.read_text(encoding="utf-8"))
        payload = data.read_bytes()
        if found["key"] != key or found["payload_sha"] != hashlib.sha256(payload).hexdigest():
            raise ValueError("CANONICAL_CONTEXT_PAYLOAD_SHA_MISMATCH")
        if found["parent_values_sha"] != spec["training_values_and_keys_sha"] or found["parent_target_sha"] != spec["target_predictor_and_B_sha"]:
            raise ValueError("CANONICAL_CONTEXT_PARENT_SHA_MISMATCH")
        if digest(found["scientific_spec"]) != digest(_context_spec(spec)):
            raise ValueError("CANONICAL_CONTEXT_SPEC_MISMATCH")
        return payload
