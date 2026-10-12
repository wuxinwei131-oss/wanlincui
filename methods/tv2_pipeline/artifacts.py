"""Freeze prediction/action artifacts before independent evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .provenance import file_sha


def freeze_scores(outdir: str | Path, county_fips: np.ndarray, year: np.ndarray,
                  scores: dict[str, np.ndarray], provenance_sha: str,
                  score_provenance_shas: dict[str, str]) -> Path:
    dest = Path(outdir)
    dest.mkdir(parents=True, exist_ok=True)
    n = len(county_fips)
    required = {f"TV2_{o}_{g}_{r}" for o in ("L1", "L2") for g in ("G0", "G1") for r in ("R0", "R1")}
    required.add("B2_L1_equalblock_selected_PB3")
    if set(scores) != required or set(score_provenance_shas) != required:
        raise ValueError("FROZEN_BENEFIT_SCORE_POLICY_SET_DRIFT")
    rows = []
    for i in range(n):
        for policy in sorted(required):
            score = float(scores[policy][i])
            if not np.isfinite(score) or len(score_provenance_shas[policy]) != 64:
                raise ValueError("NONFINITE_OR_UNPROVEN_FROZEN_BENEFIT_SCORE")
            rows.append({"county_fips": str(county_fips[i]), "year": int(year[i]), "policy": policy,
                         "benefit_score": score, "score_unit": "mg/ha PB3",
                         "score_definition": "predicted L0_PB3 - L1_PB3",
                         "action": "A1" if score > 0 else "A0",
                         "action_rule_sha": file_sha(Path(__file__).resolve().parent / "loss.py"),
                         "strategy_provenance_sha": score_provenance_shas[policy],
                         "graph_sha": provenance_sha})
    path = dest / "frozen_benefit_scores.json"
    path.write_text(json.dumps({"schema": "TV2.score.v1", "provenance_sha": provenance_sha,
                                "rows": rows}, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    manifest = dest / "FROZEN_SCORE_SHA.json"
    manifest.write_text(json.dumps({"score_path": str(path.resolve()), "score_sha256": file_sha(path),
                                    "provenance_sha": provenance_sha, "frozen": True}, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def freeze_actions(outdir: str | Path, county_fips: np.ndarray, year: np.ndarray, actions: dict[str, np.ndarray], predictions: dict[str, np.ndarray], provenance_sha: str, *, synthetic: bool, provenance_path: str | Path | None = None) -> Path:
    dest = Path(outdir)
    dest.mkdir(parents=True, exist_ok=True)
    n = len(county_fips)
    if any(len(v) != n for v in actions.values()) or any(np.asarray(v).shape != (n, 3) for v in predictions.values()):
        raise ValueError("ARTIFACT_ROW_OR_QUANTILE_SHAPE")
    rows = []
    for i in range(n):
        rows.append({"county_fips": str(county_fips[i]), "year": int(year[i]), "actions": {k: int(v[i]) for k, v in actions.items()}, "predictions": {k: np.asarray(v[i], float).tolist() for k, v in predictions.items()}})
    keys = [(r["county_fips"], r["year"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("DUPLICATE_TEST_COUNTY_YEAR")
    path = dest / "frozen_actions_predictions.json"
    path.write_text(json.dumps({"schema": "TV1R2.action.v1", "synthetic": synthetic, "provenance_sha": provenance_sha, "rows": rows}, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    manifest = {"action_path": str(path.resolve()), "action_sha256": file_sha(path), "frozen": True, "synthetic": synthetic, "provenance_sha": provenance_sha}
    if provenance_path is not None:
        graph_path = Path(provenance_path).resolve(strict=True)
        if file_sha(graph_path) != provenance_sha:
            raise ValueError("PROVENANCE_GRAPH_SHA_MISMATCH")
        manifest["provenance_path"] = str(graph_path)
    manifest_path = dest / "FROZEN_ACTION_SHA.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path
