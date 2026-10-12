"""Metadata-only nested state/year graph with complete semantic source keys."""
from __future__ import annotations

from tv2_pipeline.cache import semantic_key

STATES = ("IA", "IL", "IN", "KS", "NE", "TX")
YEAR_MIN = 2016


def plan(outer_state: str, outer_year: int, shas: dict[str, str], *, representation: str = "R1") -> dict:
    if outer_state not in (*STATES, "EXTERNAL") or outer_year not in range(2019, 2025):
        raise ValueError("UNREGISTERED_OUTER_BLOCK")
    if set(shas) != {"reference_protocol_sha", "feature_registry_sha", "base_config_sha", "fold_selection_sha", "code_sha"}:
        raise ValueError("INCOMPLETE_TASK_GRAPH_SHAS")
    # The locked geographic confirmation is pooled across external states.
    # EXTERNAL is a target-only sentinel; all six development states remain
    # legal source states and no external county enters a source meta-label.
    source = tuple(s for s in STATES if s != outer_state)
    tasks: dict[str, dict] = {}

    def add(allowed: tuple[str, ...], excluded: tuple[str, ...], label_state: str, label_year: int, geometry: str, purpose: str, validation_state: str | None = None, validation_year: int | None = None) -> None:
        item = {"allowed_states": allowed, "excluded_states": excluded, "outer_state": outer_state, "outer_year": outer_year, "validation_state": validation_state, "validation_year": validation_year, "label_state": label_state, "label_year": label_year, "predict_state": label_state, "predict_year": label_year, "geometry": geometry, "representation": representation, "effective_tree_count": 300, "execution_mode": "PLAN_ONLY_SCIENCE_DISABLED", "threads": 4, "parent_artifact_shas": [], **shas, "purpose": purpose}
        item["cache_key"] = semantic_key(item)
        tasks.setdefault(item["cache_key"], item)

    for geometry in ("G0", "G1"):
        for year in range(YEAR_MIN, outer_year):
            for label in source:
                allowed = source if geometry == "G0" else tuple(s for s in source if s != label)
                excluded = (outer_state,) if geometry == "G0" else tuple(sorted((outer_state, label)))
                add(allowed, excluded, label, year, geometry, "final_meta")
        years = tuple(range(YEAR_MIN + 1, outer_year))[-2:]
        for validation_year in years:
            for validation_state in source:
                inner = tuple(s for s in source if s != validation_state)
                add(inner, tuple(sorted((outer_state, validation_state))), validation_state, validation_year, "G1", "router_validation_pair", validation_state, validation_year)
                for year in range(YEAR_MIN, validation_year):
                    for label in inner:
                        allowed = inner if geometry == "G0" else tuple(s for s in inner if s != label)
                        excluded = tuple(sorted((outer_state, validation_state))) if geometry == "G0" else tuple(sorted((outer_state, validation_state, label)))
                        add(allowed, excluded, label, year, geometry, "router_train_meta", validation_state, validation_year)
    return {"outer_block": f"{outer_state}_{outer_year}", "tasks": list(tasks.values()), "unique_prediction_tasks": len(tasks), "registered_science_execution": "DISABLED_IN_TV1R2"}


def registered_development_blocks(registry: dict) -> tuple[str, ...]:
    if registry.get("schema_version") != "TV1.v1" or tuple(registry["development_blocks"]["states"]) != STATES or tuple(registry["development_blocks"]["years"]) != tuple(range(2019, 2025)):
        raise ValueError("FROZEN_36_BLOCK_REGISTRY_DRIFT")
    return tuple(f"{state}_{year}" for state in STATES for year in range(2019, 2025))
