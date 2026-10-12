"""Freeze lawful prior-target-year calibrations before opening target-year labels."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from qa2_model import ROOT, QA1, AUTHORITY, STATES, QS, load, dump, sha
from tv2_pipeline.loss import loss_pair, block_weights

def fixed_scores(state: str, year: int) -> pd.DataFrame:
    path = QA1 / "development36" / f"{state}_{year}" / "model_artifacts" / "frozen_benefit_scores.json"
    data = load(path)
    rows = [{"state": state, "year": year, "county_fips": str(r["county_fips"]).zfill(5),
             "benefit_score": float(r["benefit_score"])} for r in data["rows"] if r["policy"] == "TV2_L2_G0_R1"]
    result = pd.DataFrame(rows)
    assert not result.duplicated(["state", "year", "county_fips"]).any()
    return result

def prior_data(c1: Path, state: str, year: int) -> pd.DataFrame:
    cols = ["state", "county_fips", "test_year"] + [f"{a}_q{q}_mg_ha" for a in ("A0", "A1") for q in QS]
    pred = pd.read_parquet(c1 / "11_OUTER_PREDICTIONS.parquet", columns=cols,
                           filters=[("state", "==", state), ("test_year", ">=", 2019), ("test_year", "<", year)]).rename(columns={"test_year": "year"})
    hist = pd.read_parquet(c1 / "06_HISTORY_REFERENCE_PANEL.parquet", columns=["state", "county_fips", "year", "yield_mg_ha"],
                           filters=[("state", "==", state), ("year", ">=", 2019), ("year", "<", year)])
    pred["county_fips"] = pred.county_fips.astype(str).str.zfill(5)
    hist["county_fips"] = hist.county_fips.astype(str).str.zfill(5)
    pred = pred.merge(hist, on=["state", "county_fips", "year"], validate="one_to_one")
    pred = pred.loc[pred.yield_mg_ha.notna()].copy()
    assert not pred.empty and int(pred.year.max()) < year
    a0 = pred[[f"A0_q{q}_mg_ha" for q in QS]].to_numpy(float)
    a1 = pred[[f"A1_q{q}_mg_ha" for q in QS]].to_numpy(float)
    _, _, delta = loss_pair(pred.yield_mg_ha.to_numpy(float), a0, a1)
    pred["delta"] = delta
    scores = pd.concat([fixed_scores(state, y) for y in range(2019, year)], ignore_index=True)
    prior = pred[["state", "year", "county_fips", "delta"]].merge(scores,
                     on=["state", "year", "county_fips"], how="inner", validate="one_to_one")
    assert len(prior) == len(pred)
    return prior

def main():
    freeze = load(ROOT / "QA2_DESIGN_FREEZE.json")
    assert freeze["status"] == "FROZEN_BEFORE_QA2_MODEL_EVALUATION"
    authority = load(AUTHORITY)
    c1 = Path(authority["stages"]["C1L"]["path"])
    reports = []
    for state in STATES:
        for year in range(2019, 2025):
            block = f"{state}_{year}"
            path = ROOT / "model" / "LOCAL_HISTORY" / block
            checkpoint = path / "checkpoint.json"
            if checkpoint.exists():
                c = load(checkpoint)
                if c["status"] == "PASS" and sha(path / "calibrated_scores.parquet") == c["scores_sha256"]:
                    reports.append(c)
                    continue
                raise ValueError(f"LOCAL_CALIBRATION_CHECKPOINT_DRIFT:{block}")
            current = fixed_scores(state, year)
            result = current[["state", "year", "county_fips", "benefit_score"]].copy()
            report = {"block": block, "current_year_outcome_opened": False, "current_year_fit_rows": 0,
                      "source_strategy": "TV2_L2_G0_R1/WEIGHT_CORRECTED", "zero_threshold": True}
            if year == 2019:
                result["C1_INTERCEPT_score"] = np.nan
                result["C2_AFFINE_score"] = np.nan
                report.update({"C1": "NOT_ESTIMABLE_LOCAL_CALIBRATION", "C2": "NOT_ESTIMABLE_LOCAL_CALIBRATION", "prior_years": 0})
            else:
                prior = prior_data(c1, state, year)
                prior_years = sorted(int(x) for x in prior.year.unique())
                if prior_years != list(range(2019, year)):
                    raise ValueError(f"PRIOR_TARGET_YEAR_SUPPORT_GAP:{block}:{prior_years}")
                residual = prior.delta.to_numpy(float) - prior.benefit_score.to_numpy(float)
                adjustment = float(prior.assign(residual=residual).groupby("year").residual.mean().mean())
                result["C1_INTERCEPT_score"] = result.benefit_score + adjustment
                report.update({"C1": "ESTIMABLE", "C1_intercept_adjustment": adjustment, "prior_years": len(prior_years),
                               "prior_year_list": prior_years, "prior_labeled_rows": len(prior)})
                if len(prior_years) >= 2:
                    weights = block_weights(prior.state.to_numpy(), prior.year.to_numpy())
                    fit = Ridge(alpha=1., fit_intercept=True).fit(prior[["benefit_score"]].to_numpy(float),
                                                                    prior.delta.to_numpy(float), sample_weight=weights)
                    result["C2_AFFINE_score"] = fit.predict(result[["benefit_score"]].to_numpy(float))
                    report.update({"C2": "ESTIMABLE", "C2_alpha": 1.0, "C2_intercept": float(fit.intercept_),
                                   "C2_slope": float(fit.coef_[0]), "C2_prior_weight_mean": float(weights.mean())})
                else:
                    result["C2_AFFINE_score"] = np.nan
                    report["C2"] = "NOT_ESTIMABLE_LOCAL_CALIBRATION"
            path.mkdir(parents=True, exist_ok=True)
            result.to_parquet(path / "calibrated_scores.parquet", index=False)
            dump(path / "fit_report.json", report)
            c = {"status": "PASS", "block": block, "scores_sha256": sha(path / "calibrated_scores.parquet"),
                 "fit_report_sha256": sha(path / "fit_report.json"), "prior_years": report["prior_years"],
                 "bottom_source_fits": 0, "current_year_outcome_opened": False}
            dump(checkpoint, c)
            reports.append(c)
    dump(ROOT / "model" / "QA2_LOCAL_CALIBRATION_FREEZE.json",
         {"schema": "QA2.local_calibration_freeze.v1", "status": "ALL_36_SCORES_FROZEN_BEFORE_TARGET_EVALUATION",
          "blocks": reports, "bottom_source_fits": 0, "project_2025_2026_numeric_outcome_access": "NONE"})
    print(json.dumps({"status": "ALL_36_SCORES_FROZEN_BEFORE_TARGET_EVALUATION", "blocks": len(reports)}))

if __name__ == "__main__":
    main()
