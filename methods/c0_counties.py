from __future__ import annotations
from _paths import method_path, method_config, earth_engine_project

import hashlib
import json
from pathlib import Path

import ee

ROOT = method_path('corn_predictors')
RAW = ROOT / "raw" / "counties"
STATES = {"IA": ("19", 99), "IL": ("17", 102), "IN": ("18", 92),
          "NE": ("31", 93), "KS": ("20", 105), "TX": ("48", 254)}


def main() -> None:
    ee.Initialize(project=earth_engine_project())
    assert ee.Number(1).getInfo() == 1
    RAW.mkdir(parents=True, exist_ok=True)
    for state, (fips, count) in STATES.items():
        out = RAW / f"ee_tiger_2018_{state}_counties.geojson"
        if out.exists():
            data = json.loads(out.read_text(encoding="utf-8"))
        else:
            source = ee.FeatureCollection("TIGER/2018/Counties").filter(ee.Filter.eq("STATEFP", fips)).sort("GEOID")
            if source.size().getInfo() != count:
                raise RuntimeError(f"County count mismatch {state}")
            features = []
            for offset in range(0, count, 10):
                features.extend(source.toList(min(10, count-offset), offset).getInfo())
                print(f"COUNTIES {state} {len(features)}/{count}", flush=True)
            data = {"type":"FeatureCollection","source_asset":"TIGER/2018/Counties",
                    "statefp":fips,"features":[{"type":"Feature","geometry":f["geometry"],
                     "properties":{"GEOID":f["properties"]["GEOID"],"NAME":f["properties"]["NAME"]}}
                     for f in features]}
            out.write_text(json.dumps(data,separators=(",",":")),encoding="utf-8")
        ids = [f["properties"]["GEOID"] for f in data["features"]]
        if ids != sorted(set(ids)) or len(ids) != count or any(not x.startswith(fips) for x in ids):
            raise RuntimeError(f"County identity mismatch {state}")
        print(f"COUNTIES_DONE {state} {count} {out.stat().st_size} {hashlib.sha256(out.read_bytes()).hexdigest()}",flush=True)


if __name__ == "__main__":
    main()
