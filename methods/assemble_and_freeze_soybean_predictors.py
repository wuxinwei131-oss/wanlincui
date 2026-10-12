"""Assemble crop-specific August-31 panels and lock exact target keys before labels."""
from __future__ import annotations
from _paths import method_path, method_config, earth_engine_project

import csv
import hashlib
import importlib.util
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = method_path('soybean')
PRED = ROOT / "predictor"
AUDIT = ROOT / "audit"
C1L = Path(str(method_path('corn_reference')))
KEYS = ["state", "county_fips", "target_year"]
ACRE = 4046.8564224


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def longest_false(flags: list[bool]) -> int:
    top = run = 0
    for flag in flags:
        run = 0 if flag else run + 1
        top = max(top, run)
    return top


def support_quality(state: str, year: int, optical: pd.DataFrame) -> pd.DataFrame:
    weights = pd.read_parquet(PRED / "raw" / "weights" / f"{state}_{year}_MOD09A1_weights.parquet")
    counties = json.loads((PRED / "raw" / "counties" / f"ee_tiger_2018_{state}_counties.geojson").read_text(encoding="utf-8"))["features"]
    ids = [f["properties"]["GEOID"] for f in counties]
    summary = weights.groupby("county_fips", sort=True).agg(nominal_crop_weight_m2=("nominal_weight_q255", "sum"),
                                                           mod09_nonzero_grid_cells=("grid_row", "count"))
    expected = int(optical.composite_start.nunique())
    if expected != 18:
        raise RuntimeError(f"MOD09A1 composite count drift {state} {year}: {expected}")
    rows = []
    for fips in ids:
        nominal = float(summary.loc[fips, "nominal_crop_weight_m2"]) if fips in summary.index else 0.0
        cells = int(summary.loc[fips, "mod09_nonzero_grid_cells"]) if fips in summary.index else 0
        op = optical.loc[optical.county_fips.eq(fips)].sort_values("composite_start")
        if len(op) != expected:
            raise RuntimeError(f"Incomplete county composites {fips} {year}")
        frac = op.valid_weight_fraction.to_numpy(float)
        valid = op.valid_crop_weight_m2.to_numpy(float) > 0
        rows.append({"state": state, "county_fips": fips, "target_year": year,
                     "prior_soybean_effective_acres": nominal / ACRE, "nominal_crop_weight_m2": nominal,
                     "mod09_nonzero_grid_cells": cells, "expected_eligible_composites": expected,
                     "composites_positive_valid_support": int(valid.sum()),
                     "valid_composite_fraction": float(valid.mean()),
                     "valid_weight_fraction_min": float(np.nanmin(frac)) if np.isfinite(frac).any() else math.nan,
                     "valid_weight_fraction_median": float(np.nanmedian(frac)) if np.isfinite(frac).any() else math.nan,
                     "valid_weight_fraction_max": float(np.nanmax(frac)) if np.isfinite(frac).any() else math.nan,
                     "longest_missing_composite_run": longest_false(valid.tolist()),
                     "PRIMARY_CROP_SUPPORT": nominal / ACRE >= 500,
                     "STRICT_CROP_SUPPORT": nominal / ACRE >= 1000})
    return pd.DataFrame(rows)


def main() -> None:
    if (AUDIT / "TV5_MODEL_ANALYSIS_CODE_FREEZE.json").exists() or any(
        (ROOT / "chronology" / str(year) / "TV5_TARGET_LABELS.parquet").exists()
        for year in range(2019, 2025)
    ):
        raise RuntimeError("Outcome-blind exact-key assembly cannot run after model freeze or target label access")
    completion = json.loads((AUDIT / "TV5_PREDICTOR_BUILD_COMPLETION.json").read_text(encoding="utf-8"))
    if completion["status"] != "COMPLETE":
        raise RuntimeError("Predictor construction not complete; target labels sealed")
    plan_path = AUDIT / "TV5_PREDICTOR_JOB_PLAN.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    jobs = [(x["state"], int(x["year"])) for x in plan["jobs"]]
    spec = importlib.util.spec_from_file_location("frozen_c1l_features", C1L / "c1l_features.py")
    assert spec and spec.loader
    c1l = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c1l)
    frozen_groups_path = C1L / "C1L_MODEL_FEATURE_GROUPS.json"
    frozen_groups = json.loads(frozen_groups_path.read_text(encoding="utf-8"))
    expected_groups = {k: [x.replace("prior_corn_effective_acres", "prior_soybean_effective_acres") for x in v]
                       for k, v in frozen_groups.items()}
    optics, weathers, supports, sources = [], [], [], []
    for state, year in jobs:
        optical_path = PRED / "state_year" / f"{state}_{year}_optical.parquet"
        weather_path = PRED / "state_year" / f"{state}_{year}_weather.parquet"
        optical_metrics_path = PRED / "state_year" / f"{state}_{year}_optical_metrics.json"
        weather_metrics_path = PRED / "state_year" / f"{state}_{year}_weather_metrics.json"
        for stage, data_path, metrics_path in (("optical", optical_path, optical_metrics_path),
                                               ("weather", weather_path, weather_metrics_path)):
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if (metrics.get("state") != state or metrics.get("target_year") != year
                    or metrics.get("parquet_sha256") != sha(data_path)):
                raise RuntimeError(f"Resumed {stage} parquet differs from executed output metrics {state} {year}")
        optical = pd.read_parquet(optical_path)
        weather = pd.read_parquet(weather_path)
        for product in ("MOD09A1", "GRIDMET"):
            weight_path = PRED / "raw" / "weights" / f"{state}_{year}_{product}_weights.parquet"
            weight_result = PRED / "raw" / "weights" / f"{state}_{year}_{product}_result.json"
            info = json.loads(weight_result.read_text(encoding="utf-8"))
            if info["weights_sha256"] != sha(weight_path) or info["prior_cdl_years"] != [year - 3, year - 2, year - 1]:
                raise RuntimeError(f"Soybean weight content/chronology drift {state} {year} {product}")
        for prior_year in range(year - 3, year):
            cdl = json.loads((PRED / f"C0_CDL_SOURCE_{state}_{prior_year}.json").read_text(encoding="utf-8"))
            if cdl.get("soybean_class") != 5 or cdl["year"] != prior_year:
                raise RuntimeError(f"Crop-specific CDL source identity drift {state} {prior_year}")
        support = support_quality(state, year, optical)
        if not optical.composite_end.str[-5:].le("08-31").all() or not weather.gridmet_date_end.str[-5:].eq("08-31").all():
            raise RuntimeError(f"August-31 violation {state} {year}")
        if set(support.county_fips) != set(weather.county_fips) or set(support.county_fips) != set(optical.county_fips):
            raise RuntimeError(f"County predictor join mismatch {state} {year}")
        optics.append(optical); weathers.append(weather); supports.append(support)
        sources.append({"state": state, "target_year": year, "optical_sha256": sha(optical_path),
                        "weather_sha256": sha(weather_path),
                        "optical_metrics_sha256": sha(optical_metrics_path),
                        "weather_metrics_sha256": sha(weather_metrics_path),
                        "MOD09A1_weight_sha256": sha(PRED / "raw" / "weights" / f"{state}_{year}_MOD09A1_weights.parquet"),
                        "GRIDMET_weight_sha256": sha(PRED / "raw" / "weights" / f"{state}_{year}_GRIDMET_weights.parquet"),
                        "county_count": len(support), "prior_CDL_years": [year - 3, year - 2, year - 1]})
    optical = pd.concat(optics, ignore_index=True)
    weather = pd.concat(weathers, ignore_index=True)
    support = pd.concat(supports, ignore_index=True)
    if support.duplicated(KEYS).any() or weather.duplicated(KEYS).any() or optical.duplicated(KEYS + ["composite_start"]).any():
        raise RuntimeError("Predictor key duplication")
    signal_rows = [dict(zip(KEYS, key), **c1l.signal_stats(part)) for key, part in optical.groupby(KEYS, sort=True)]
    signals = pd.DataFrame(signal_rows)
    weather_features = [c for c in weather.columns if c not in KEYS + ["nominal_crop_weight_m2", "effective_prior_soybean_area_acres", "gridmet_daily_precip_complete", "gridmet_date_start", "gridmet_date_end"] and not c.endswith("valid_weight_fraction")]
    quality = ["composites_positive_valid_support", "valid_composite_fraction", "longest_missing_composite_run", "valid_weight_fraction_median", "valid_weight_fraction_min", "prior_soybean_effective_acres"] + [f"q_{x}_valid_fraction" for x in c1l.MONTHS.values()]
    optical_features = [c for c in signals if c.startswith("opt_")]
    groups = {"A0": weather_features, "AQ": weather_features + quality,
              "AS": weather_features + optical_features, "A1": weather_features + optical_features + quality}
    if groups != expected_groups:
        raise RuntimeError("Soybean feature groups differ from frozen C1L method except crop-name substitution")
    panel = support[KEYS + ["prior_soybean_effective_acres", "composites_positive_valid_support", "valid_composite_fraction", "valid_weight_fraction_median", "valid_weight_fraction_min", "longest_missing_composite_run", "PRIMARY_CROP_SUPPORT", "STRICT_CROP_SUPPORT"]].merge(weather[KEYS + weather_features + ["gridmet_daily_precip_complete"]], on=KEYS, validate="one_to_one").merge(signals, on=KEYS, validate="one_to_one")
    panel["A1_available"] = panel.composites_positive_valid_support.gt(0)
    panel = panel.sort_values(KEYS).reset_index(drop=True)
    if any(x in c.lower() for c in panel for x in ("yield", "benefit", "residual", "pb3", "harvested")):
        raise RuntimeError("Outcome-like field in predictor panel")
    panel_path = ROOT / "TV5_SOYBEAN_PREDICTOR_PANEL.parquet"
    panel.to_parquet(panel_path, index=False, compression="zstd")
    group_path = ROOT / "TV5_SOYBEAN_FEATURE_GROUPS.json"
    group_path.write_text(json.dumps(groups, indent=2) + "\n", encoding="utf-8")
    candidate_paths = (AUDIT / "TV5_DEVELOPMENT_HISTORICAL_CANDIDATES.csv", AUDIT / "TV5_EXTERNAL_CAPPED_CANDIDATES.csv")
    candidates = pd.concat([pd.read_csv(p, dtype={"state": str, "county_fips": str, "ERS_region": str}) for p in candidate_paths], ignore_index=True)
    if candidates.duplicated("county_fips").any():
        raise RuntimeError("Development/external candidate overlap")
    keys = candidates[["state", "county_fips", "ERS_region"]].merge(panel, on=["state", "county_fips"], how="inner", validate="one_to_many")
    keys["predictor_eligible"] = keys.PRIMARY_CROP_SUPPORT & keys.A1_available & keys.gridmet_daily_precip_complete
    keys["predictor_failure_reason"] = [";".join(reason for failed, reason in ((not bool(r.PRIMARY_CROP_SUPPORT), "PRIOR_SOYBEAN_LT_500_ACRES"),
                                                                                     (not bool(r.A1_available), "NO_VALID_OPTICAL_COMPOSITE"),
                                                                                     (not bool(r.gridmet_daily_precip_complete), "GRIDMET_DAILY_PRECIP_INCOMPLETE")) if failed) or "NONE"
                                        for r in keys.itertuples(index=False)]
    key_columns = ["ERS_region", "state", "county_fips", "target_year", "prior_soybean_effective_acres",
                   "composites_positive_valid_support", "valid_composite_fraction", "valid_weight_fraction_median",
                   "longest_missing_composite_run", "PRIMARY_CROP_SUPPORT", "A1_available",
                   "gridmet_daily_precip_complete", "predictor_eligible", "predictor_failure_reason"]
    target_audit = keys.loc[keys.target_year.between(2019, 2024), key_columns].sort_values(["ERS_region", "state", "county_fips", "target_year"])
    expected_target_keys = {(str(r.state), str(r.county_fips), year)
                            for r in candidates.itertuples(index=False) for year in range(2019, 2025)}
    actual_target_keys = {(str(r.state), str(r.county_fips), int(r.target_year))
                          for r in target_audit.itertuples(index=False)}
    if actual_target_keys != expected_target_keys or len(target_audit) != len(expected_target_keys):
        missing = expected_target_keys - actual_target_keys
        extra = actual_target_keys - expected_target_keys
        raise RuntimeError(f"Target predictor candidate-key coverage incomplete: missing={len(missing)} extra={len(extra)}")
    support_path = AUDIT / "TV5_PREDICTOR_SUPPORT_AUDIT.csv"
    target_audit.to_csv(support_path, index=False, float_format="%.10g")
    # A region qualifies only with >=20 historically selected counties whose
    # predictor support is complete for all six target years. No failed key is replaced.
    dev_states = set(plan["development_states"])
    development = target_audit.loc[target_audit.state.isin(dev_states)].copy()
    development_full_counties = (development.groupby(["state", "county_fips"])
                                 .predictor_eligible.sum().eq(6).reset_index(name="full_six_year_support"))
    development_support_counts = {state: int(part.full_six_year_support.sum())
                                  for state, part in development_full_counties.groupby("state")}
    development_support_counts = {state: development_support_counts.get(state, 0) for state in sorted(dev_states)}
    external = target_audit.loc[~target_audit.state.isin(dev_states)].copy()
    full_counties = external.groupby(["ERS_region", "county_fips"]).predictor_eligible.sum().eq(6).reset_index(name="full_six_year_support")
    qualifying = sorted({r for r, part in full_counties.groupby("ERS_region") if int(part.full_six_year_support.sum()) >= 20})
    if sum(count >= 20 for count in development_support_counts.values()) < 4:
        status = "SOYBEAN_SOURCE_DOMAIN_INFEASIBLE"
    elif any(count < 20 for count in development_support_counts.values()):
        status = "DEVELOPMENT_SELECTION_REVISION_REQUIRED_BEFORE_LABELS"
    elif len(qualifying) < 3:
        status = "SOYBEAN_EXTERNAL_CONFIRMATION_INFEASIBLE"
    else:
        status = "PASS"
    selected_external = external.loc[external.ERS_region.isin(qualifying) & external.predictor_eligible, key_columns]
    selected_dev = target_audit.loc[target_audit.state.isin(dev_states) & target_audit.predictor_eligible, key_columns]
    dev_path = AUDIT / "TV5_DEVELOPMENT_EXACT_KEYS.csv"
    ext_path = AUDIT / "TV5_EXTERNAL_EXACT_KEYS.csv"
    selected_dev.to_csv(dev_path, index=False, float_format="%.10g")
    selected_external.to_csv(ext_path, index=False, float_format="%.10g")
    freeze = {"schema": "TV5.preoutcome_predictor_and_exact_key_freeze.v1", "utc": datetime.now(timezone.utc).isoformat(),
              "status": status, "development_states": sorted(dev_states), "external_regions": qualifying,
              "development_complete_six_year_counties_by_state": development_support_counts,
              "external_complete_six_year_counties_by_region": {r: int(part.full_six_year_support.sum()) for r, part in full_counties.groupby("ERS_region")},
              "development_exact_key_count": len(selected_dev), "external_exact_key_count": len(selected_external),
              "development_exact_keys_path": str(dev_path), "development_exact_keys_sha256": sha(dev_path),
              "external_exact_keys_path": str(ext_path), "external_exact_keys_sha256": sha(ext_path),
              "predictor_panel_path": str(panel_path), "predictor_panel_sha256": sha(panel_path),
              "predictor_support_audit_sha256": sha(support_path), "feature_groups_sha256": sha(group_path),
              "frozen_C1L_feature_groups_sha256": sha(frozen_groups_path), "predictor_job_plan_sha256": sha(plan_path),
              "source_state_year_sha256": sources, "2019_2024_target_outcome_accessed": False,
              "project_2025_2026_outcome_accessed": False}
    (AUDIT / "TV5_PREOUTCOME_EXACT_KEY_FREEZE.json").write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "development_keys": len(selected_dev), "external_keys": len(selected_external),
                      "external_regions": qualifying}))
    if status != "PASS":
        raise RuntimeError(f"Soybean pre-outcome predictor/domain gate {status}; labels remain sealed")


if __name__ == "__main__":
    main()
