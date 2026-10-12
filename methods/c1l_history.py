from __future__ import annotations
from _paths import method_path, method_config, earth_engine_project

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor

ROOT = method_path('corn_reference')
CONVERSION = 0.0627676647


def huber_fit(history: pd.DataFrame, target_year: int) -> float:
    x = (history["year"].to_numpy(dtype=float) - 2000.0).reshape(-1, 1)
    y = history["yield_mg_ha"].to_numpy(dtype=float)
    model = HuberRegressor()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    return float(model.predict(np.array([[target_year - 2000.0]]))[0]), sum(issubclass(w.category, ConvergenceWarning) for w in captured)


def main() -> None:
    target = pd.read_parquet(ROOT / "04_COUNTY_YEAR_TARGET_PANEL.parquet")
    freeze = json.loads((ROOT / "TARGET_FREEZE_LITE_2000_2024.json").read_text(encoding="utf-8"))
    assert len(target) == freeze["canonical_county_grain_rows"] == 12690
    target["yield_mg_ha"] = target["yield_bu_acre"] * CONVERSION
    main = target.loc[target.year.between(2012, 2024)].copy().sort_values(["year", "state", "county_fips"])
    assert len(main) == 6023
    by_county = {key: group.sort_values("year") for key, group in target.groupby("county_fips", sort=False)}
    by_state = {key: group.sort_values("year") for key, group in target.groupby("state", sort=False)}
    state_cache: dict[tuple[str, int], tuple[pd.DataFrame, float, int]] = {}
    pool_cache: dict[int, tuple[pd.DataFrame, float, int]] = {}
    records = []
    for row in main.itertuples(index=False):
        year = int(row.year)
        county_history = by_county[row.county_fips]
        county_history = county_history.loc[county_history.year.lt(year)]
        if len(county_history) >= 5 and int(county_history.year.max() - county_history.year.min()) >= 4:
            history = county_history
            reference_level = "SAME_COUNTY_HUBER"
            B, warnings_count = huber_fit(history, year)
        else:
            key = (row.state, year)
            if key not in state_cache:
                state_history = by_state[row.state].loc[by_state[row.state].year.lt(year)]
                if len(state_history):
                    predicted, warnings_count = huber_fit(state_history, year)
                    state_cache[key] = (state_history, predicted, warnings_count)
            if key in state_cache:
                history, B, warnings_count = state_cache[key]
                reference_level = "SAME_STATE_HUBER"
            else:
                if year not in pool_cache:
                    pool_history = target.loc[target.year.lt(year)]
                    if not len(pool_history):
                        raise RuntimeError(f"No chronology-safe historical reference for {row.county_fips} {year}")
                    predicted, warnings_count = huber_fit(pool_history, year)
                    pool_cache[year] = (pool_history, predicted, warnings_count)
                history, B, warnings_count = pool_cache[year]
                reference_level = "SIX_STATE_POOLED_HUBER"
        min_year, max_year = int(history.year.min()), int(history.year.max())
        assert max_year < year
        records.append({"state": row.state, "county_fips": row.county_fips, "year": year,
                        "yield_bu_acre": float(row.yield_bu_acre), "yield_mg_ha": float(row.yield_mg_ha),
                        "B_history_mg_ha": B, "residual_mg_ha": float(row.yield_mg_ha) - B,
                        "reference_level": reference_level, "history_rows": len(history),
                        "same_county_prior_rows": len(county_history), "min_history_year": min_year,
                        "max_history_year": max_year, "history_huber_convergence_warnings": warnings_count})
    panel = pd.DataFrame(records).sort_values(["state", "year", "county_fips"]).reset_index(drop=True)
    assert len(panel) == 6023 and not panel.duplicated(["county_fips", "year"]).any()
    panel.to_parquet(ROOT / "06_HISTORY_REFERENCE_PANEL.parquet", index=False, compression="zstd")
    leakage = panel[["state", "county_fips", "year", "reference_level", "history_rows", "min_history_year", "max_history_year", "history_huber_convergence_warnings"]].copy()
    leakage["history_leakage"] = leakage.max_history_year.ge(leakage.year)
    leakage.to_csv(ROOT / "07_HISTORY_LEAKAGE_AUDIT.csv", index=False)
    print(json.dumps({"history_rows": len(panel), "leakage_rows": int(leakage.history_leakage.sum()),
                      "reference_levels": panel.reference_level.value_counts().to_dict(),
                      "convergence_warning_rows": int(panel.history_huber_convergence_warnings.gt(0).sum()),
                      "yield_unit": "Mg/ha"}, indent=2))


if __name__ == "__main__":
    main()
