"""Retrieve only frozen-key year-t NASS soybean labels after action SHA freeze."""
from __future__ import annotations
from _paths import method_path, method_config, earth_engine_project

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = method_path('soybean')
LITE = "https://www.nass.usda.gov/Quick_Stats/Lite"
LB_KG = 0.45359237
ACRE_HA = 0.40468564224
CONVERSION = 60 * LB_KG / ACRE_HA / 1000


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def number(value: object) -> float | None:
    try:
        result = float(str(value).replace(",", ""))
        return result if result >= 0 else None
    except (ValueError, TypeError):
        return None


def request_retry(session: requests.Session, method: str, url: str, **kwargs) -> requests.Response:
    for attempt in range(1, 4):
        try:
            response = session.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(5 * attempt)
    raise AssertionError("Unreachable NASS request retry state")


def historical_locations(selected: set[str]) -> dict[str, str]:
    out = defaultdict(set)
    source = ROOT / "sources" / "NASS_LITE_SOYBEAN_2000_2018_BY_STATE"
    for path in source.glob("*_SOYBEAN_2000_2018.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        for row in raw["items"]:
            fips = str(row["STATE_ANSI"]) + str(row.get("COUNTY_ANSI", ""))
            if fips in selected:
                out[fips].add(row["LOCATION_DESC"])
    if set(out) != selected or any(len(names) != 1 for names in out.values()):
        raise RuntimeError("Frozen county location names ambiguous or missing")
    return {fips: next(iter(names)) for fips, names in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", required=True, type=int, choices=range(2019, 2025))
    args = parser.parse_args()
    year = args.year
    year_dir = ROOT / "chronology" / str(year)
    action_path = year_dir / "TV5_YEAR_ACTION_PREDICTION_FREEZE.json"
    code_path = ROOT / "audit" / "TV5_MODEL_ANALYSIS_CODE_FREEZE.json"
    exact_path = ROOT / "audit" / "TV5_PREOUTCOME_EXACT_KEY_FREEZE.json"
    if not all(x.exists() for x in (action_path, code_path, exact_path)):
        raise RuntimeError("Pre-result freezes absent; target outcome retrieval forbidden")
    safe_probe = json.loads((ROOT / "audit" / "TV5_SAFE2018_LARGEST_STATE_QUERY_PROBE.json").read_text(encoding="utf-8"))
    if (safe_probe["status"] != "PASS"
            or not safe_probe["joined_form_decodes_exact_string_and_returns_only_selected_counties"]):
        raise RuntimeError("Pre-result NASS exact-county selector probe not passed")
    if year > 2019:
        prior_path = ROOT / "chronology" / str(year - 1) / "TV5_YEAR_EVALUATION_SEAL.json"
        if not prior_path.exists() or json.loads(prior_path.read_text(encoding="utf-8"))["status"] != "SEALED":
            raise RuntimeError("Previous target year not independently evaluated and sealed")
    code = json.loads(code_path.read_text(encoding="utf-8"))
    if sha(Path(__file__)) != next(x["sha256"] for x in code["code_files"] if x["path"] == str(Path(__file__))):
        raise RuntimeError("Label broker code SHA drift")
    if sha(ROOT / "audit" / "TV5_SAFE2018_LARGEST_STATE_QUERY_PROBE.json") != code["safe_2018_NASS_selector_probe_sha256"]:
        raise RuntimeError("Pre-result NASS selector probe SHA drift")
    historical_selector_path = ROOT / "audit" / "TV5_SAFE_HISTORICAL_NASS_SELECTOR_SUPPORT.json"
    if (sha(historical_selector_path) != code["safe_historical_NASS_selector_support_sha256"]
            or json.loads(historical_selector_path.read_text(encoding="utf-8"))["status"] != "PASS"):
        raise RuntimeError("Pre-result historical NASS county selector support SHA drift")
    all_state_probe_path = ROOT / "audit" / "TV5_SAFE2018_ALL_STATE_SELECTOR_PROBE.json"
    if (sha(all_state_probe_path) != code["safe_2018_all_state_NASS_selector_probe_sha256"]
            or json.loads(all_state_probe_path.read_text(encoding="utf-8"))["status"] != "PASS"):
        raise RuntimeError("Pre-result 2018 all-state NASS selector probe SHA drift")
    action = json.loads(action_path.read_text(encoding="utf-8"))
    if action["status"] != "FROZEN_BEFORE_TARGET_LABEL_ACCESS" or action["year"] != year:
        raise RuntimeError("Current-year actions not sealed")
    if action["preoutcome_code_freeze_sha256"] != sha(code_path) or action["exact_key_freeze_sha256"] != sha(exact_path):
        raise RuntimeError("Model/keys changed after action freeze")
    for entry in action["outputs"]:
        if (sha(Path(entry["prediction_path"])) != entry["prediction_sha256"]
                or sha(Path(entry["model_report_path"])) != entry["model_report_sha256"]):
            raise RuntimeError("Action/prediction/model-report SHA drift before target outcomes")
    analysis = json.loads((ROOT / "TV5_SOYBEAN_ANALYSIS_FREEZE.json").read_text(encoding="utf-8"))
    states = analysis["development_states"]
    expected_tasks = {("SOURCE_DOMAIN_TEMPORAL", state, "Z2") for state in states}
    expected_tasks |= {("UNSEEN_GEOGRAPHY", state, z) for state in states for z in ("Z1", "Z2")}
    expected_tasks.add(("EXTERNAL_CONFIRMATION", "EXTERNAL_ALL", "Z2"))
    actual_tasks = {(entry["condition"], entry["state"], entry["Z"]) for entry in action["outputs"]}
    if actual_tasks != expected_tasks or len(action["outputs"]) != len(expected_tasks):
        raise RuntimeError("Not all fixed tasks are frozen before target label retrieval")
    label_path = year_dir / "TV5_TARGET_LABELS.parquet"
    manifest_path = year_dir / "TV5_TARGET_LABEL_ACCESS_MANIFEST.json"
    if label_path.exists() or manifest_path.exists():
        if not (label_path.exists() and manifest_path.exists()):
            raise RuntimeError("Partial target label retrieval must be audited")
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior["labels_sha256"] != sha(label_path) or prior["year"] != year:
            raise RuntimeError("Existing label SHA drift")
        print(json.dumps({"year": year, "status": "LABELS_ALREADY_VERIFIED", "available": prior["label_available"]}))
        return
    dev = pd.read_csv(ROOT / "audit" / "TV5_DEVELOPMENT_EXACT_KEYS.csv", dtype={"state": str, "county_fips": str})
    ext = pd.read_csv(ROOT / "audit" / "TV5_EXTERNAL_EXACT_KEYS.csv", dtype={"state": str, "county_fips": str})
    keys = pd.concat([dev, ext], ignore_index=True).loc[lambda x: x.target_year.eq(year), ["state", "county_fips", "target_year"]]
    if keys.duplicated(["county_fips", "target_year"]).any() or keys.empty:
        raise RuntimeError("Frozen target key identity invalid")
    fips_to_location = historical_locations(set(keys.county_fips))
    state_names = {r["STUSAB"]: r["STATE_NAME"].upper() for r in csv.DictReader((ROOT / "sources" / "CENSUS_state_fips_reference.txt").open(encoding="utf-8"), delimiter="|")}
    raw_dir = year_dir / "nass_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    retrieved = {}
    source_files = []
    session = requests.Session()
    for state, part in keys.groupby("state", sort=True):
        locations = sorted(fips_to_location[x] for x in part.county_fips)
        joined_locations = " , ".join(locations)
        params = {"SECTOR_DESC": "CROPS", "GROUP_DESC": "FIELD CROPS", "COMMODITY_DESC": "SOYBEANS",
                  "REPORT_NAME": "6", "YEAR": str(year), "AGG_LEVEL_DESC": "COUNTY", "STATE_NAME": state_names[state],
                  "LOCATION_DESC": joined_locations}
        enc = request_retry(session, "POST", "https://www.nass.usda.gov/qs/uuid/encode/", data=params, timeout=90)
        uuid = enc.text.strip().strip('"')
        if len(uuid) != 36:
            raise RuntimeError(f"NASS target query UUID invalid {state} {year}")
        decoded = request_retry(session, "GET", "https://www.nass.usda.gov/qs/uuid/decode/" + uuid, timeout=60).json()
        if (decoded.get("YEAR") != [str(year)] or decoded.get("STATE_NAME") != [state_names[state]]
                or decoded.get("COMMODITY_DESC") != ["SOYBEANS"]
                or decoded.get("AGG_LEVEL_DESC") != ["COUNTY"]
                or decoded.get("LOCATION_DESC") != [joined_locations]):
            raise RuntimeError(f"NASS target query decoded outside frozen year/crop/level {state}")
        response = request_retry(session, "GET", LITE + "/ajax.php?" + uuid, timeout=180)
        raw = response.content
        out = raw_dir / f"{state}_SOYBEAN_{year}.json"
        out.write_bytes(raw)
        items = response.json().get("items", [])
        allowed = set(part.county_fips)
        for item in items:
            if (int(item["YEAR"]) != year or item.get("COMMODITY") != "SOYBEANS"
                    or item.get("REFERENCE_PERIOD_DESC") != "YEAR"):
                raise RuntimeError("Forbidden target year/crop in NASS response")
            fips = str(item["STATE_ANSI"]) + str(item.get("COUNTY_ANSI", ""))
            if not fips.startswith(str(part.county_fips.iloc[0])[:2]):
                raise RuntimeError("NASS target response state FIPS mismatch")
            if fips not in allowed:
                if len(str(item.get("COUNTY_ANSI", ""))) == 3:
                    raise RuntimeError("NASS target response exceeded frozen county selector")
                # Non-county combined aggregates never enter numeric parsing.
                continue
            value = number(item.get("YIELD in BU / ACRE"))
            if value is None:
                continue
            if fips in retrieved and retrieved[fips] != value:
                raise RuntimeError(f"Ambiguous target label {fips} {year}")
            retrieved[fips] = value
        source_files.append({"state": state, "path": str(out), "sha256": sha(out),
                             "query_uuid": uuid, "requested_counties": len(allowed), "response_rows": len(items)})
    labels = keys.sort_values(["state", "county_fips"]).reset_index(drop=True)
    labels["year"] = year
    labels["yield_bu_acre"] = labels.county_fips.map(retrieved)
    labels["yield_mg_ha"] = labels.yield_bu_acre * CONVERSION
    labels["label_status"] = labels.yield_mg_ha.notna().map({True: "AVAILABLE", False: "LABEL_NOT_AVAILABLE"})
    labels.to_parquet(label_path, index=False, compression="zstd")
    manifest = {"schema": "TV5.target_label_access_after_action_freeze.v1", "year": year,
                "access_utc": datetime.now(timezone.utc).isoformat(), "action_freeze_sha256": sha(action_path),
                "code_freeze_sha256": sha(code_path), "exact_key_freeze_sha256": sha(exact_path),
                "requested_exact_keys": len(labels), "label_available": int(labels.yield_mg_ha.notna().sum()),
                "label_not_available": int(labels.yield_mg_ha.isna().sum()),
                "labels_path": str(label_path), "labels_sha256": sha(label_path),
                "source_files": source_files, "no_replacement_keys": True,
                "project_2025_2026_outcome_accessed": False}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"year": year, "status": "LABELS_RETRIEVED_AFTER_ACTION_FREEZE",
                      "available": manifest["label_available"], "missing": manifest["label_not_available"]}))


if __name__ == "__main__":
    main()
