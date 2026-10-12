"""Deterministic TV5 domain selection using pre-2019 soybean history only."""
from __future__ import annotations
from _paths import method_path, method_config, earth_engine_project

import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = method_path('soybean')
AUDIT = ROOT / "audit"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source = AUDIT / "TV5_HISTORICAL_CANDIDATES.csv"
    rows = list(csv.DictReader(source.open(encoding="utf-8", newline="")))
    eligible = [r for r in rows if r["historical_eligible"] == "True"]
    by_pair = defaultdict(list)
    for r in eligible:
        by_pair[(r["ERS_region"], r["state"])].append(r)
    by_region = defaultdict(list)
    for (region, state), part in by_pair.items():
        if len(part) < 20:
            continue
        by_region[region].append({"region": region, "state": state, "state_fips": min(x["state_fips"] for x in part),
                                  "eligible_counties_in_region": len(part),
                                  "historical_area_acres_sum": sum(float(x["total_2000_2018_harvested_area_acres"]) for x in part),
                                  "median_county_area_acres": statistics.median(float(x["median_harvested_area_acres"]) for x in part)})
    winners = [sorted(part, key=lambda x: (-x["historical_area_acres_sum"], x["state_fips"]))[0]
               for _, part in sorted(by_region.items())]
    # Where more than six regions qualify, take the six with largest historical
    # eligible-area support; this rule is fixed before any target-year outcome.
    ranked = sorted(winners, key=lambda x: (-x["historical_area_acres_sum"], x["region"], x["state_fips"]))
    selected = ranked[:6]
    if len(set(x["state"] for x in selected)) != len(selected):
        raise RuntimeError("One state selected for multiple ERS regions; unique-state assignment not defined")
    states = sorted(x["state"] for x in selected)
    dev = [r for r in eligible if r["state"] in states]
    external = [r for r in eligible if r["state"] not in states]
    region_external = defaultdict(list)
    for r in external:
        region_external[r["ERS_region"]].append(r)
    capped_external = []
    for region, part in sorted(region_external.items()):
        if len(part) < 20:
            continue
        capped_external.extend(sorted(part, key=lambda x: (-float(x["median_harvested_area_acres"]), x["county_fips"]))[:40])
    def write_csv(name: str, content: list[dict]) -> dict:
        path = AUDIT / name
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(content)
        return {"path": str(path), "sha256": sha(path), "counties": len(content)}
    development_file = write_csv("TV5_DEVELOPMENT_HISTORICAL_CANDIDATES.csv", dev)
    external_file = write_csv("TV5_EXTERNAL_HISTORICAL_CANDIDATES.csv", external)
    external_cap_file = write_csv("TV5_EXTERNAL_CAPPED_CANDIDATES.csv", capped_external)
    freeze = {"schema": "TV5.preoutcome_historical_domain_selection.v1",
              "selection_input": str(source), "selection_input_sha256": sha(source),
              "selection_rule": "Within each ERS region, states with >=20 historically eligible counties qualify; choose largest sum of 2000-2018 eligible-county harvested acres, ties by state FIPS. Choose up to six distinct region winners by descending support, ties by ERS code then state FIPS.",
              "selected_region_state_pairs": selected, "development_states": states,
              "development_historical_candidates": development_file,
              "external_historical_candidates": external_file,
              "external_40_per_region_pre_predictor_cap": external_cap_file,
              "external_cap_rule": "Every nondevelopment region with >=20 historically eligible counties; top 40 by <=2018 median harvested soybean acres descending, FIPS ascending, fixed before predictor and target outcomes; failed predictor keys are not replaced.",
              "external_historical_region_counts_after_dev_exclusion": {k: len(v) for k, v in sorted(region_external.items())},
              "predictor_support_complete": False, "exact_keys_frozen": False,
              "2019_2024_target_outcome_accessed": False, "project_2025_2026_outcome_accessed": False}
    (AUDIT / "TV5_PREOUTCOME_DOMAIN_SELECTION.json").write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"development_states": states, "development_counties": len(dev),
                      "external_region_counts": freeze["external_historical_region_counts_after_dev_exclusion"]}))


if __name__ == "__main__":
    main()
