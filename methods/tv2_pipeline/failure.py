"""Task failures remain explicit and propagate to aggregate readiness."""
from __future__ import annotations


def task_status(source_rows: int, meta_rows: int, validation_rows: int, *, source_failed: bool = False) -> dict:
    reasons = []
    if source_failed:
        reasons.append("SOURCE_FIT_FAILED")
    if source_rows == 0:
        reasons.append("NO_SOURCE_ROWS")
    if meta_rows == 0:
        reasons.append("NO_META_LABELS")
    if validation_rows == 0:
        reasons.append("NO_SOURCE_VALIDATION_ROWS")
    return {"status": "NOT_ESTIMABLE" if reasons else "ESTIMABLE", "reasons": reasons}


def propagate(task_results: list[dict]) -> dict:
    blocked = [r for r in task_results if r["status"] != "ESTIMABLE"]
    return {"status": "NOT_ESTIMABLE" if blocked else "COMPLETE", "blocked_tasks": len(blocked), "required_tasks": len(task_results)}
