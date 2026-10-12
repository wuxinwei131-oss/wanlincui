"""Make the legal <=2018 soybean yield history for frozen candidate counties."""
from __future__ import annotations
from _paths import method_path, method_config, earth_engine_project

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = method_path('soybean')
SOURCE = ROOT / "sources" / "NASS_LITE_SOYBEAN_2000_2018_BY_STATE"
LB_KG = 0.45359237
ACRE_HA = 0.40468564224
SOYBEAN_BU_LB = 60
CONVERSION = SOYBEAN_BU_LB * LB_KG / ACRE_HA / 1000


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
    except (TypeError, ValueError):
        return None


def main() -> None:
    paths = (ROOT / "audit" / "TV5_DEVELOPMENT_HISTORICAL_CANDIDATES.csv",
             ROOT / "audit" / "TV5_EXTERNAL_CAPPED_CANDIDATES.csv")
    selected = {r["county_fips"] for path in paths for r in csv.DictReader(path.open(encoding="utf-8", newline=""))}
    values = {}
    sources = []
    for path in sorted(SOURCE.glob("*_SOYBEAN_2000_2018.json")):
        meta_path = path.with_name(path.name.replace(".json", ".metadata.json"))
        if not meta_path.exists() or json.loads(meta_path.read_text(encoding="utf-8"))["sha256"] != sha(path):
            raise RuntimeError(f"Historical soybean source hash failed {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        sources.append({"path": str(path), "sha256": sha(path)})
        for row in raw["items"]:
            year = int(row["YEAR"])
            if not 2000 <= year <= 2018 or row["COMMODITY"] != "SOYBEANS":
                raise RuntimeError("Forbidden year or crop in safe history")
            fips = str(row["STATE_ANSI"]) + str(row.get("COUNTY_ANSI", ""))
            if len(fips) != 5 or not fips.isdigit():
                continue
            yield_bu = number(row.get("YIELD in BU / ACRE"))
            if yield_bu is None:
                continue
            area = number(row.get("AREA HARVESTED in ACRES"))
            key = (fips, year)
            val = {"state": path.name[:2], "county_fips": fips, "year": year,
                   "yield_bu_acre": yield_bu, "yield_mg_ha": yield_bu * CONVERSION,
                   "harvested_area_acres": area}
            if key in values and values[key] != val:
                raise RuntimeError(f"Ambiguous historical soybean value {key}")
            values[key] = val
    frame = pd.DataFrame(values.values()).sort_values(["state", "county_fips", "year"]).reset_index(drop=True)
    if frame.year.max() > 2018 or not selected <= set(frame.county_fips) or frame.duplicated(["county_fips", "year"]).any():
        raise RuntimeError("Safe soybean historical panel identity invalid")
    out = ROOT / "TV5_SAFE_SOYBEAN_HISTORY_2000_2018.parquet"
    frame.to_parquet(out, index=False, compression="zstd")
    manifest = {"schema": "TV5.safe_soybean_history_panel.v1", "years": [2000, 2018],
                "selected_candidate_counties": len(selected), "reference_pool_counties": int(frame.county_fips.nunique()),
                "history_rows": len(frame),
                "yield_conversion_mg_ha_per_bu_acre": CONVERSION,
                "standard_bushel_lb": 60,
                "standard_bushel_source": "https://www-tx.ers.usda.gov/data-products/oil-crops-yearbook/documentation",
                "history_path": str(out), "history_sha256": sha(out), "source_files": sources,
                "2019_2024_target_outcome_accessed": False, "project_2025_2026_outcome_accessed": False}
    (ROOT / "audit" / "TV5_SAFE_SOYBEAN_HISTORY_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(frame), "reference_pool_counties": int(frame.county_fips.nunique()),
                      "selected_candidates": len(selected), "conversion": CONVERSION}))


if __name__ == "__main__":
    main()
