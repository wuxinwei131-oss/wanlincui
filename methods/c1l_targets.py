from __future__ import annotations
from _paths import method_path, method_config, earth_engine_project

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = method_path('corn_reference')
C0 = ROOT.parent / "CRP_C0_CLEAN_PREDICTOR_BUILD_20260912_160900"
INPUT = ROOT / "inputs" / "nass_lite"
STATE_BY_ANSI = {"19": "IA", "17": "IL", "18": "IN", "20": "KS", "31": "NE", "48": "TX"}
EXPECTED_SHA = {
    "IA": "d8b0224b724ad291ce44e73d5fe06f8494a3b9926d24ed2db87bfd2031540d65",
    "IL": "e2fa3d461b1a03411cceb220c9cf3a4b4819f4ca7b7ff21e3d43150b5a119441",
    "IN": "a50e162d11a96403eb089d065d9964337a1d0148e6d2207916f2459f4dd786c1",
    "KS": "cc06217fa92d165bfb2cdf0003903bb3dcf7f459d9a7e76c0716cc12eb3e4196",
    "NE": "3c78b125619956cf7401bca1fffe37b70c3a4b6058b753ac10c7d383ca478d0e",
    "TX": "d8563579aaa40cb37c828498e97e5ee353436c4c6684ce46d18acf9adfd10094",
}
SCHEMA = ["YEAR", "LOCATION", "STATE ANSI", "ASD CODE", "COUNTY ANSI", "REFERENCE PERIOD", "COMMODITY", "AREA HARVESTED in ACRES", "PRODUCTION in BU", "YIELD in BU / ACRE", "AREA PLANTED in ACRES", "PRODUCTION in TONS", "YIELD in TONS / ACRE"]
ANCHORS = {"IA": (5916, 2391, 1203), "IL": (4882, 2353, 1144), "IN": (4373, 2082, 1034), "KS": (4949, 2038, 919), "NE": (4834, 2023, 959), "TX": (4113, 1803, 764)}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")


def main() -> None:
    assert (ROOT / "00A_PREDICTOR_AUTHORITY_VERIFICATION.md").is_file()
    assert (ROOT / "00B_TARGET_AND_MODEL_RULE_FREEZE.json").is_file()
    freeze = json.loads((C0 / "PREDICTOR_FREEZE_BEFORE_OUTCOMES.json").read_text(encoding="utf-8"))
    assert freeze["complete_state_years"] == 78 and freeze["county_year_key_count"] == 9685
    assert all(sha(C0 / name) == digest for name, digest in freeze["predictor_file_sha256"].items())
    files = sorted(INPUT.glob("*.csv"))
    assert len(files) == 6
    ledgers, audit_frames, aggregate_frames, targets = [], [], [], []
    seen_states = set()
    for path in files:
        raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        assert raw.columns.tolist() == SCHEMA, f"Lite schema mismatch: {path}"
        codes = set(raw["STATE ANSI"].str.strip())
        assert len(codes) == 1 and next(iter(codes)) in STATE_BY_ANSI
        state_ansi = next(iter(codes))
        state = STATE_BY_ANSI[state_ansi]
        assert state not in seen_states
        seen_states.add(state)
        digest = sha(path)
        assert digest == EXPECTED_SHA[state], f"SOURCE_HASH_DIFFERENT_REVIEW_REQUIRED: {state}"
        year = number(raw["YEAR"])
        assert year.notna().all() and year.mod(1).eq(0).all()
        assert year.min() == 2000 and year.max() == 2024 and year.nunique() == 25
        assert not year.isin([2025, 2026]).any()
        assert raw["REFERENCE PERIOD"].eq("YEAR").all()
        county = raw["COUNTY ANSI"].str.strip()
        county_valid = county.str.fullmatch(r"\d{1,3}").fillna(False)
        grain = raw["COMMODITY"].eq("CORN, GRAIN")
        yield_num = number(raw["YIELD in BU / ACRE"])
        area_num = number(raw["AREA HARVESTED in ACRES"])
        prod_num = number(raw["PRODUCTION in BU"])
        canonical = grain & county_valid & yield_num.notna()
        grain_county = grain & county_valid
        main = canonical & year.between(2012, 2024)
        assert int(grain_county.sum()) == int(canonical.sum()), f"Non-numeric independent grain yield: {state}"
        assert area_num[main].notna().all(), f"Main-period harvested area missing: {state}"
        assert (len(raw), int(grain_county.sum()), int(main.sum())) == ANCHORS[state], f"C1L_SOURCE_AUDIT_MISMATCH: {state}"
        keys = raw.loc[canonical, "STATE ANSI"].str.zfill(2) + county.loc[canonical].str.zfill(3) + "|" + year.loc[canonical].astype(int).astype(str)
        assert not keys.duplicated().any(), f"Duplicate county-year target: {state}"
        raw_audit = raw.copy()
        raw_audit.insert(0, "source_row_number", np.arange(2, len(raw) + 2))
        raw_audit.insert(0, "source_file_sha256", digest)
        raw_audit.insert(0, "identified_state", state)
        raw_audit["county_ansi_parseable"] = county_valid
        raw_audit["grain_exact"] = grain
        raw_audit["yield_numeric"] = yield_num.notna()
        raw_audit["canonical_independent_grain_target"] = canonical
        raw_audit["main_target_year"] = year.between(2012, 2024)
        raw_audit["historical_yield_with_missing_area"] = canonical & year.lt(2012) & area_num.isna()
        audit_frames.append(raw_audit)
        aggregates = raw_audit.loc[~county_valid].copy()
        aggregates["noncounty_reason"] = "COUNTY_ANSI_BLANK_OR_UNPARSEABLE_NO_SINGLE_COUNTY_MAPPING"
        aggregate_frames.append(aggregates)
        target = pd.DataFrame({
            "year": year.loc[canonical].astype(int),
            "state_ansi": raw.loc[canonical, "STATE ANSI"].str.zfill(2),
            "county_ansi": county.loc[canonical].str.zfill(3),
            "county_fips": raw.loc[canonical, "STATE ANSI"].str.zfill(2) + county.loc[canonical].str.zfill(3),
            "state": state,
            "location": raw.loc[canonical, "LOCATION"],
            "yield_bu_acre": yield_num.loc[canonical].astype(float),
            "harvested_acres": area_num.loc[canonical].astype(float),
            "production_bu": prod_num.loc[canonical].astype(float),
            "source_file_sha256": digest,
            "historical_yield_with_missing_area": (year.loc[canonical].lt(2012) & area_num.loc[canonical].isna()).astype(bool),
        })
        targets.append(target)
        ledgers.append({"state": state, "state_ansi": state_ansi, "file": path.name, "sha256": digest, "bytes": path.stat().st_size,
                        "source": "USDA_NASS_QUICK_STATS_LITE_MANUAL_EXPORT", "sector": "CROPS", "group": "FIELD CROPS", "commodity_selection": "CORN",
                        "view": "Acreage, Yield, Production and Price", "geographic_level": "COUNTY", "min_year": 2000, "max_year": 2024,
                        "unique_years": 25, "all_rows": len(raw), "grain_independent_county_rows": int(grain_county.sum()),
                        "main_numeric_yield_rows": int(main.sum()), "reference_period_all_YEAR": True,
                        "schema_13_columns_pass": True, "target_year_2025_2026_present": False})
    assert seen_states == set(EXPECTED_SHA)
    source_ledger = pd.DataFrame(ledgers).sort_values("state")
    target_panel = pd.concat(targets, ignore_index=True).sort_values(["state", "year", "county_fips"]).reset_index(drop=True)
    assert len(target_panel) == 12690 and int(target_panel.year.between(2012, 2024).sum()) == 6023
    assert not target_panel.duplicated(["county_fips", "year"]).any()
    assert target_panel.year.between(2000, 2024).all()
    support = pd.read_parquet(C0 / "08_SUPPORT_QUALITY_PANEL.parquet")
    support = support.rename(columns={"target_year": "year"})
    assert len(support) == 9685 and not support.duplicated(["county_fips", "year"]).any()
    target_keys = target_panel.loc[target_panel.year.between(2012, 2024), ["county_fips", "year", "yield_bu_acre", "harvested_acres"]]
    availability = support[["state", "county_fips", "year", "PRIMARY_CROP_SUPPORT", "STRICT_CROP_SUPPORT", "composites_positive_valid_support", "valid_composite_fraction", "prior_corn_effective_acres"]].merge(target_keys, on=["county_fips", "year"], how="left", validate="one_to_one", indicator=True)
    availability["target_status"] = np.where(availability["_merge"].eq("both"), "NUMERIC_CANONICAL_LITE_TARGET", "NO_INDEPENDENT_COUNTY_TARGET_IN_LITE_EXPORT")
    availability["A1_available"] = availability["composites_positive_valid_support"].gt(0)
    availability["primary_numeric"] = availability["PRIMARY_CROP_SUPPORT"] & availability["target_status"].eq("NUMERIC_CANONICAL_LITE_TARGET")
    availability["strict_numeric"] = availability["STRICT_CROP_SUPPORT"] & availability["target_status"].eq("NUMERIC_CANONICAL_LITE_TARGET")
    availability["paired_primary_numeric"] = availability["primary_numeric"] & availability["A1_available"]
    outer = []
    for year in range(2019, 2025):
        for state in ("IA", "IL", "IN", "NE", "KS", "TX"):
            block = availability.loc[availability.state.eq(state) & availability.year.eq(year)]
            dev = availability.loc[availability.state.ne(state) & availability.year.lt(year) & availability.year.ge(2012)]
            outer.append({"held_state": state, "test_year": year, "block_id": f"{state}_{year}",
                          "raw_numeric_target_rows": int(target_panel.state.eq(state).mul(target_panel.year.eq(year)).sum()),
                          "frozen_predictor_rows": len(block), "numeric_target_matches": int(block.target_status.eq("NUMERIC_CANONICAL_LITE_TARGET").sum()),
                          "primary_numeric_rows": int(block.primary_numeric.sum()), "strict_numeric_rows": int(block.strict_numeric.sum()),
                          "paired_A1_available_rows": int(block.paired_primary_numeric.sum()),
                          "development_paired_rows": int(dev.paired_primary_numeric.sum()),
                          "development_max_year": year - 1, "inner_validation_year": year - 1,
                          "inner_training_max_year": year - 2, "held_state_in_development": False,
                          "future_year_in_development": False})
    outer_frame = pd.DataFrame(outer)
    assert len(outer_frame) == 36 and outer_frame.raw_numeric_target_rows.min() >= 42
    source_ledger.to_csv(ROOT / "01_NASS_LITE_SOURCE_LEDGER.csv", index=False)
    pd.concat(audit_frames, ignore_index=True).to_csv(ROOT / "02_TARGET_DEFINITION_AUDIT.csv", index=False)
    pd.concat(aggregate_frames, ignore_index=True).to_csv(ROOT / "03_NONCOUNTY_AGGREGATE_LEDGER.csv", index=False)
    target_panel.to_parquet(ROOT / "04_COUNTY_YEAR_TARGET_PANEL.parquet", index=False, compression="zstd")
    availability.drop(columns=["_merge"]).to_csv(ROOT / "05_TARGET_AVAILABILITY_LEDGER.csv", index=False)
    outer_frame.to_csv(ROOT / "08_OUTER_FOLD_REGISTRY.csv", index=False)
    target_freeze = {"source": "USDA_NASS_QUICK_STATS_LITE_MANUAL_EXPORT", "contains_2025": False, "contains_2026": False,
                     "source_hashes": {row.state: row.sha256 for row in source_ledger.itertuples()},
                     "canonical_filter": "COMMODITY == 'CORN, GRAIN' AND parseable nonblank COUNTY ANSI AND expected STATE ANSI AND REFERENCE PERIOD == 'YEAR' AND 2000<=YEAR<=2024 AND numeric YIELD in BU / ACRE",
                     "county_fips": "STATE ANSI zfill(2)+COUNTY ANSI zfill(3)", "historical_reference_years": [2000, 2024],
                     "main_model_years": [2012, 2024], "main_numeric_target_rows": 6023, "canonical_county_grain_rows": 12690,
                     "target_parquet_sha256": sha(ROOT / "04_COUNTY_YEAR_TARGET_PANEL.parquet"),
                     "rules_freeze_sha256": sha(ROOT / "00B_TARGET_AND_MODEL_RULE_FREEZE.json"),
                     "suppression_code_inference": "FORBIDDEN; missing label NO_INDEPENDENT_COUNTY_TARGET_IN_LITE_EXPORT"}
    (ROOT / "TARGET_FREEZE_LITE_2000_2024.json").write_text(json.dumps(target_freeze, indent=2), encoding="utf-8")
    print(json.dumps({"status": "TARGET_FREEZE_PASS", "source_rows": int(source_ledger.all_rows.sum()), "canonical_rows": len(target_panel),
                      "main_rows": 6023, "predictor_keys": len(availability), "target_matches": int(availability.target_status.eq("NUMERIC_CANONICAL_LITE_TARGET").sum()),
                      "missing_independent": int(availability.target_status.ne("NUMERIC_CANONICAL_LITE_TARGET").sum()),
                      "primary_numeric": int(availability.primary_numeric.sum()), "strict_numeric": int(availability.strict_numeric.sum()),
                      "paired": int(availability.paired_primary_numeric.sum()), "outer_blocks": len(outer_frame),
                      "outer_ge20_paired": int(outer_frame.paired_A1_available_rows.ge(20).sum()),
                      "min_outer_paired": int(outer_frame.paired_A1_available_rows.min())}, indent=2))


if __name__ == "__main__":
    main()
