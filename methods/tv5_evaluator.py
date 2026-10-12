"""Independent fixed-key evaluator; reads year-t labels only after prediction SHA seal."""
from __future__ import annotations
from _paths import method_path, method_config, earth_engine_project

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = method_path('soybean')
QS = (10, 50, 90)
QFRACS = (.1, .5, .9)
KEYS = ["state", "county_fips", "year"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pinball(y: np.ndarray, pred: np.ndarray, q: float) -> np.ndarray:
    err = y - pred
    return np.maximum(q * err, (q - 1) * err)


def evaluate_prediction(pred: pd.DataFrame, labels: pd.DataFrame, entry: dict,
                        exact: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pred.duplicated(KEYS).any() or not set(pred.action) <= {"A0", "A1"}:
        raise RuntimeError("Frozen action prediction keys/values invalid")
    merged = pred.merge(exact[KEYS + ["ERS_region"]], on=KEYS, how="left", validate="one_to_one")
    if merged.ERS_region.isna().any():
        raise RuntimeError("Prediction outside exact frozen key support")
    merged = merged.merge(labels[KEYS + ["yield_mg_ha", "label_status"]], on=KEYS, how="left", validate="one_to_one")
    if merged.label_status.isna().any():
        raise RuntimeError("Exact prediction key absent from label broker ledger")
    merged["block"] = merged.ERS_region if entry["condition"] == "EXTERNAL_CONFIRMATION" else merged.state
    out = merged.loc[merged.label_status.eq("AVAILABLE")].copy()
    if out.empty:
        return out, pd.DataFrame([{"condition": entry["condition"], "state": entry["state"], "Z": entry["Z"],
                                   "year": int(pred.year.iloc[0]), "block": block, "exact_keys": len(part),
                                   "labels_available": 0, "status": "LABEL_NOT_AVAILABLE"}
                                  for block, part in merged.groupby("block")])
    y = out.yield_mg_ha.to_numpy(float)
    for action in (("A0", "A1", "AS") if entry["condition"] == "EXTERNAL_CONFIRMATION" else ("A0", "A1")):
        for q_int, q in zip(QS, QFRACS):
            out[f"{action}_q{q_int}_pinball"] = pinball(y, out[f"{action}_q{q_int}_mg_ha"].to_numpy(float), q)
        out[f"{action}_PB3"] = out[[f"{action}_q{q}_pinball" for q in QS]].mean(axis=1)
    choose_a1 = out.action.eq("A1").to_numpy()
    out["routing_PB3"] = np.where(choose_a1, out.A1_PB3, out.A0_PB3)
    for q in QS:
        out[f"routing_q{q}_pinball"] = np.where(choose_a1, out[f"A1_q{q}_pinball"], out[f"A0_q{q}_pinball"])
    out["true_benefit_delta"] = out.A0_PB3 - out.A1_PB3
    out["benefit_absolute_error"] = (out.benefit_score - out.true_benefit_delta).abs()
    out["zero_score_absolute_error"] = out.true_benefit_delta.abs()
    out["source_constant_absolute_error"] = (out.source_constant_score - out.true_benefit_delta).abs()
    out["non_tie"] = out.true_benefit_delta.ne(0)
    out["sign_correct"] = np.where(out.non_tie, out.action.eq(np.where(out.true_benefit_delta.gt(0), "A1", "A0")), np.nan)
    out["always_A1_sign_correct"] = np.where(out.non_tie, out.true_benefit_delta.gt(0), np.nan)
    out["false_switch"] = out.action.eq("A1") & out.true_benefit_delta.lt(0)
    out["missed_rescue"] = out.action.eq("A0") & out.true_benefit_delta.gt(0)
    out["source_static_PB3"] = np.where(out.source_static_action.eq("A1"), out.A1_PB3, out.A0_PB3)
    oracle_a1 = out.true_benefit_delta.gt(0).to_numpy()
    for q in QS:
        a0q = out[f"A0_q{q}_pinball"].to_numpy(float)
        a1q = out[f"A1_q{q}_pinball"].to_numpy(float)
        oracle_q = np.where(oracle_a1, a1q, a0q)
        out[f"oracle_q{q}_pinball"] = oracle_q
        out[f"routing_oracle_q{q}_regret"] = out[f"routing_q{q}_pinball"] - oracle_q
        out[f"false_switch_q{q}_regret"] = np.where(out.false_switch, a1q - a0q, 0.)
        out[f"missed_rescue_q{q}_regret"] = np.where(out.missed_rescue, a0q - a1q, 0.)
        out[f"tie_q{q}_regret"] = np.where(out.true_benefit_delta.eq(0) & out.action.eq("A1"), a1q - a0q, 0.)
    rows = []
    for block, part in merged.groupby("block", sort=True):
        valid = out.loc[out.block.eq(block)]
        row = {"condition": entry["condition"], "state": entry["state"], "Z": entry["Z"],
               "year": int(part.year.iloc[0]), "block": block, "exact_keys": len(part),
               "labels_available": len(valid), "label_missing": len(part) - len(valid),
               "status": "EVALUABLE" if len(valid) else "LABEL_NOT_AVAILABLE"}
        if len(valid):
            for name in ("routing_PB3", "A0_PB3", "A1_PB3", "source_static_PB3",
                         "benefit_absolute_error", "zero_score_absolute_error", "source_constant_absolute_error",
                         "sign_correct", "always_A1_sign_correct", "false_switch", "missed_rescue"):
                row[name] = float(valid[name].mean())
            if "AS_PB3" in valid:
                row["AS_PB3"] = float(valid.AS_PB3.mean())
            row["routing_minus_A1_PB3"] = row["routing_PB3"] - row["A1_PB3"]
            row["routing_minus_AS_PB3"] = row["routing_PB3"] - row["AS_PB3"] if "AS_PB3" in row else np.nan
            for q in QS:
                row[f"routing_minus_A1_q{q}"] = float((valid[f"routing_q{q}_pinball"] - valid[f"A1_q{q}_pinball"]).mean())
                for name in ("routing_oracle", "false_switch", "missed_rescue", "tie"):
                    row[f"{name}_q{q}_regret"] = float(valid[f"{name}_q{q}_regret"].mean())
                if not np.isclose(row[f"routing_oracle_q{q}_regret"],
                                  sum(row[f"{name}_q{q}_regret"] for name in ("false_switch", "missed_rescue", "tie")),
                                  atol=1e-9):
                    raise RuntimeError("Component-wise oracle switch regret identity failed")
            row["mean_true_benefit_delta"] = float(valid.true_benefit_delta.mean())
            row["mean_predicted_benefit_score"] = float(valid.benefit_score.mean())
        rows.append(row)
    out["condition"] = entry["condition"]
    out["Z"] = entry["Z"]
    out["task_state"] = entry["state"]
    return out, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", required=True, type=int, choices=range(2019, 2025))
    args = parser.parse_args()
    year = args.year
    year_dir = ROOT / "chronology" / str(year)
    action_path = year_dir / "TV5_YEAR_ACTION_PREDICTION_FREEZE.json"
    labels_path = year_dir / "TV5_TARGET_LABELS.parquet"
    access_path = year_dir / "TV5_TARGET_LABEL_ACCESS_MANIFEST.json"
    code_path = ROOT / "audit" / "TV5_MODEL_ANALYSIS_CODE_FREEZE.json"
    for path in (action_path, labels_path, access_path, code_path):
        if not path.exists():
            raise RuntimeError(f"Independent evaluator prerequisite missing {path}")
    code = json.loads(code_path.read_text(encoding="utf-8"))
    own_sha = next(x["sha256"] for x in code["code_files"] if x["path"] == str(Path(__file__)))
    if sha(Path(__file__)) != own_sha:
        raise RuntimeError("Independent evaluator code SHA drift")
    action = json.loads(action_path.read_text(encoding="utf-8"))
    access = json.loads(access_path.read_text(encoding="utf-8"))
    if access["action_freeze_sha256"] != sha(action_path) or access["labels_sha256"] != sha(labels_path):
        raise RuntimeError("Label access/prediction chronology SHA mismatch")
    analysis_freeze = json.loads((ROOT / "TV5_SOYBEAN_ANALYSIS_FREEZE.json").read_text(encoding="utf-8"))
    states = analysis_freeze["development_states"]
    expected_tasks = {("SOURCE_DOMAIN_TEMPORAL", state, "Z2") for state in states}
    expected_tasks |= {("UNSEEN_GEOGRAPHY", state, z) for state in states for z in ("Z1", "Z2")}
    expected_tasks.add(("EXTERNAL_CONFIRMATION", "EXTERNAL_ALL", "Z2"))
    actual_tasks = {(entry["condition"], entry["state"], entry["Z"]) for entry in action["outputs"]}
    if actual_tasks != expected_tasks or len(action["outputs"]) != len(expected_tasks):
        raise RuntimeError("Frozen action task registry incomplete or duplicated")
    seal_path = year_dir / "TV5_YEAR_EVALUATION_SEAL.json"
    if seal_path.exists():
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        if seal["status"] == "SEALED" and all(sha(Path(x["path"])) == x["sha256"] for x in seal["outputs"]):
            print(json.dumps({"year": year, "status": "EVALUATION_ALREADY_SEALED"}))
            return
        raise RuntimeError("Existing evaluation seal failed verification")
    labels = pd.read_parquet(labels_path)
    dev = pd.read_csv(ROOT / "audit" / "TV5_DEVELOPMENT_EXACT_KEYS.csv", dtype={"state": str, "county_fips": str, "ERS_region": str})
    ext = pd.read_csv(ROOT / "audit" / "TV5_EXTERNAL_EXACT_KEYS.csv", dtype={"state": str, "county_fips": str, "ERS_region": str})
    exact = pd.concat([dev, ext], ignore_index=True).rename(columns={"target_year": "year"})
    exact = exact.loc[exact.year.eq(year), KEYS + ["ERS_region"]]
    if set(map(tuple, exact[KEYS].to_numpy())) != set(map(tuple, labels[KEYS].to_numpy())):
        raise RuntimeError("Independent evaluator exact-key mismatch")
    eval_dir = year_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    outputs = []
    all_blocks = []
    for entry in action["outputs"]:
        pred_path = Path(entry["prediction_path"])
        if (sha(pred_path) != entry["prediction_sha256"]
                or sha(Path(entry["model_report_path"])) != entry["model_report_sha256"]):
            raise RuntimeError("Frozen prediction or model report changed before evaluator read")
        pred = pd.read_parquet(pred_path)
        county, blocks = evaluate_prediction(pred, labels, entry, exact)
        stem = f"{entry['condition']}_{entry['state']}_{entry['Z']}"
        county_path = eval_dir / f"{stem}_county.csv"
        block_path = eval_dir / f"{stem}_blocks.csv"
        county.to_csv(county_path, index=False, float_format="%.10g")
        blocks.to_csv(block_path, index=False, float_format="%.10g")
        all_blocks.append(blocks)
        outputs += [{"path": str(county_path), "sha256": sha(county_path)},
                    {"path": str(block_path), "sha256": sha(block_path)}]
    summary_path = eval_dir / "TV5_YEAR_BLOCK_SUMMARY.csv"
    pd.concat(all_blocks, ignore_index=True).to_csv(summary_path, index=False, float_format="%.10g")
    outputs.append({"path": str(summary_path), "sha256": sha(summary_path)})
    # Make the just-evaluated year legal history only after all fixed-policy
    # metrics have been written. B was frozen in the pre-label predictions.
    source_predictions = []
    for entry in action["outputs"]:
        if (entry["condition"] == "SOURCE_DOMAIN_TEMPORAL" and entry["Z"] == "Z2") or entry["condition"] == "EXTERNAL_CONFIRMATION":
            p = pd.read_parquet(entry["prediction_path"], columns=KEYS + ["B_history_mg_ha", "max_history_year"])
            source_predictions.append(p)
    B = pd.concat(source_predictions, ignore_index=True)
    if B.duplicated(KEYS).any() or set(map(tuple, B[KEYS].to_numpy())) != set(map(tuple, exact[KEYS].to_numpy())):
        raise RuntimeError("Frozen B reference exact-key mismatch")
    revealed = labels.loc[labels.label_status.eq("AVAILABLE"), KEYS + ["yield_bu_acre", "yield_mg_ha"]].merge(B, on=KEYS, validate="one_to_one")
    revealed["residual_mg_ha"] = revealed.yield_mg_ha - revealed.B_history_mg_ha
    prior_path = ROOT / "chronology" / "TV5_REVEALED_PRIOR_HISTORY.parquet"
    if year > 2019:
        prior_snapshot = ROOT / "chronology" / str(year - 1) / f"TV5_REVEALED_HISTORY_THROUGH_{year - 1}.parquet"
        prior_seal_path = ROOT / "chronology" / str(year - 1) / "TV5_YEAR_EVALUATION_SEAL.json"
        prior_seal = json.loads(prior_seal_path.read_text(encoding="utf-8"))
        if sha(prior_snapshot) != prior_seal["revealed_prior_history_snapshot_sha256"]:
            raise RuntimeError("Prior-year immutable revealed-history SHA drift")
        prior = pd.read_parquet(prior_snapshot)
        if not prior.empty and prior.year.max() >= year:
            raise RuntimeError("Future/current outcome in prior-year snapshot")
        revealed = pd.concat([prior, revealed], ignore_index=True)
    revealed = revealed.sort_values(KEYS).reset_index(drop=True)
    snapshot_path = year_dir / f"TV5_REVEALED_HISTORY_THROUGH_{year}.parquet"
    if snapshot_path.exists():
        if not pd.read_parquet(snapshot_path).equals(revealed):
            raise RuntimeError("Interrupted year snapshot differs from recomputed legal history")
    else:
        revealed.to_parquet(snapshot_path, index=False, compression="zstd")
    # Seal the immutable year-specific snapshot. The current-history pointer is
    # intentionally mutable when the next evaluation year is revealed.
    revealed.to_parquet(prior_path, index=False, compression="zstd")
    if sha(prior_path) != sha(snapshot_path):
        raise RuntimeError("Current-history pointer differs from year snapshot")
    outputs.append({"path": str(snapshot_path), "sha256": sha(snapshot_path)})
    seal = {"schema": "TV5.independent_year_evaluation_seal.v1", "status": "SEALED", "year": year,
            "sealed_utc": datetime.now(timezone.utc).isoformat(), "action_freeze_sha256": sha(action_path),
            "label_access_manifest_sha256": sha(access_path), "labels_sha256": sha(labels_path),
            "fixed_prediction_outputs": len(action["outputs"]), "exact_keys": len(exact),
            "labels_available": int(labels.yield_mg_ha.notna().sum()),
            "label_not_available": int(labels.yield_mg_ha.isna().sum()),
            "revealed_prior_history_snapshot_sha256": sha(snapshot_path), "outputs": outputs,
            "policy_reselection": False, "project_2025_2026_outcome_accessed": False}
    seal_path.write_text(json.dumps(seal, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"year": year, "status": "SEALED", "labels_available": seal["labels_available"],
                      "label_not_available": seal["label_not_available"]}))


if __name__ == "__main__":
    main()
