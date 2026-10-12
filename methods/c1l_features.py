from __future__ import annotations
from _paths import method_path, method_config, earth_engine_project

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = method_path('corn_reference')
C0 = ROOT.parent / "CRP_C0_CLEAN_PREDICTOR_BUILD_20260912_160900"
SIGNALS = {
    "red_reflectance": "red", "nir_reflectance": "nir", "blue_reflectance": "blue",
    "green_reflectance": "green", "swir_1240nm_reflectance": "swir1240",
    "swir_1640nm_reflectance": "swir1640", "swir_2130nm_reflectance": "swir2130",
    "ndvi": "ndvi", "evi2": "evi2", "nirv": "nirv", "ndmi": "ndmi",
}
MONTHS = {4: "apr", 5: "may", 6: "jun", 7: "jul", 8: "aug"}
KEYS = ["state", "county_fips", "target_year"]


def signal_stats(group: pd.DataFrame) -> dict[str, float]:
    group = group.sort_values("composite_start")
    date = pd.to_datetime(group["composite_start"])
    midpoint_doy = date.dt.dayofyear.to_numpy(dtype=float) + 3.5
    months = date.dt.month.to_numpy(dtype=int)
    positive = group["valid_crop_weight_m2"].to_numpy(dtype=float) > 0
    result = {}
    for month, label in MONTHS.items():
        result[f"q_{label}_valid_fraction"] = float(np.mean(positive[months == month]))
    for original, short in SIGNALS.items():
        values = group[original].to_numpy(dtype=float)
        valid = positive & np.isfinite(values)
        for month, label in MONTHS.items():
            subset = values[valid & (months == month)]
            result[f"opt_{short}_{label}_mean"] = float(np.mean(subset)) if len(subset) else np.nan
        x, y = midpoint_doy[valid], values[valid]
        result[f"opt_{short}_season_mean"] = float(np.mean(y)) if len(y) else np.nan
        result[f"opt_{short}_season_max"] = float(np.max(y)) if len(y) else np.nan
        result[f"opt_{short}_last_valid"] = float(y[-1]) if len(y) else np.nan
        result[f"opt_{short}_slope_per_day"] = float(np.polyfit(x, y, 1)[0]) if len(y) >= 2 and x[-1] > x[0] else np.nan
        result[f"opt_{short}_auc_per_day_span"] = float(np.trapezoid(y, x) / (x[-1] - x[0])) if len(y) >= 2 and x[-1] > x[0] else np.nan
    return result


def main() -> None:
    optical = pd.read_parquet(C0 / "06_OPTICAL_COMPOSITE_FEATURES.parquet")
    weather = pd.read_parquet(C0 / "07_WEATHER_FEATURES.parquet")
    support = pd.read_parquet(C0 / "08_SUPPORT_QUALITY_PANEL.parquet")
    assert len(optical) == 174330 and len(weather) == len(support) == 9685
    assert optical["composite_start"].str[-5:].ge("04-01").all()
    assert optical["composite_end"].str[-5:].le("08-31").all()
    assert weather["gridmet_date_start"].str[-5:].eq("01-01").all()
    assert weather["gridmet_date_end"].str[-5:].eq("08-31").all()
    signal_rows = []
    for key, group in optical.groupby(KEYS, sort=True):
        assert len(group) == 18
        signal_rows.append(dict(zip(KEYS, key), **signal_stats(group)))
    signals = pd.DataFrame(signal_rows)
    assert len(signals) == 9685 and not signals.duplicated(KEYS).any()
    weather_features = [col for col in weather.columns if col not in KEYS + ["nominal_crop_weight_m2", "effective_prior_corn_area_acres", "gridmet_daily_precip_complete", "gridmet_date_start", "gridmet_date_end"] and not col.endswith("valid_weight_fraction")]
    support_columns = ["prior_corn_effective_acres", "composites_positive_valid_support", "valid_composite_fraction", "valid_weight_fraction_median", "valid_weight_fraction_min", "longest_missing_composite_run", "PRIMARY_CROP_SUPPORT", "STRICT_CROP_SUPPORT"]
    panel = support[KEYS + support_columns].merge(weather[KEYS + weather_features], on=KEYS, validate="one_to_one").merge(signals, on=KEYS, validate="one_to_one")
    panel["A1_available"] = panel["composites_positive_valid_support"].gt(0)
    panel = panel.sort_values(KEYS).reset_index(drop=True)
    assert len(panel) == 9685 and panel.A1_available.eq(panel.composites_positive_valid_support.gt(0)).all()
    quality_features = ["composites_positive_valid_support", "valid_composite_fraction", "longest_missing_composite_run", "valid_weight_fraction_median", "valid_weight_fraction_min", "prior_corn_effective_acres"] + [f"q_{x}_valid_fraction" for x in MONTHS.values()]
    signal_features = [col for col in signals.columns if col not in KEYS and col.startswith("opt_")]
    assert len(signal_features) == 110 and len(quality_features) == 11
    feature_registry = []
    for original, short in SIGNALS.items():
        unit = "unitless" if short in ("ndvi", "evi2", "ndmi") else "reflectance_scaled" if "reflectance" in original else "index"
        for month, label in MONTHS.items():
            feature_registry.append({"feature": f"opt_{short}_{label}_mean", "source_signal": original, "summary": f"mean of valid eligible composites starting in month {month}", "unit": unit, "date_rule": "complete support ends by target-year Aug 31"})
        for suffix, definition in (("season_mean", "mean of valid Apr-Aug composites"), ("season_max", "maximum valid Apr-Aug composite"), ("last_valid", "latest valid Apr-Aug composite"), ("slope_per_day", "OLS slope versus composite midpoint day-of-year"), ("auc_per_day_span", "trapezoidal AUC divided by observed midpoint-day span")):
            feature_registry.append({"feature": f"opt_{short}_{suffix}", "source_signal": original, "summary": definition, "unit": unit, "date_rule": "complete support ends by target-year Aug 31"})
    pd.DataFrame(feature_registry).to_csv(ROOT / "09_OPTICAL_SUMMARY_FEATURE_REGISTRY.csv", index=False)
    feature_systems = {"A0": weather_features, "AQ": weather_features + quality_features,
                       "AS": weather_features + signal_features,
                       "A1": weather_features + signal_features + quality_features}
    model_registry = []
    for system, columns in feature_systems.items():
        for feature in columns:
            role = "weather" if feature in weather_features else "quality" if feature in quality_features else "optical_signal"
            model_registry.append({"system": system, "feature": feature, "role": role,
                                   "source": "C0 frozen gridMET" if role == "weather" else "C0 frozen MOD09 QA/support" if role == "quality" else "C0 frozen MOD09 18-composite signal summary",
                                   "may_use_harvested_area": False, "may_use_target_yield": False})
    pd.DataFrame(model_registry).to_csv(ROOT / "10_MODEL_FEATURE_REGISTRY.csv", index=False)
    panel.to_parquet(ROOT / "C1L_MODEL_FEATURE_PANEL.parquet", index=False, compression="zstd")
    (ROOT / "C1L_MODEL_FEATURE_GROUPS.json").write_text(json.dumps(feature_systems, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(panel), "weather_features": len(weather_features), "signal_features": len(signal_features),
                      "quality_features": len(quality_features), "system_feature_counts": {key: len(value) for key, value in feature_systems.items()},
                      "A1_available_rows": int(panel.A1_available.sum())}, indent=2))


if __name__ == "__main__":
    main()
