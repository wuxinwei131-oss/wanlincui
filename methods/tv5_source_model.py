"""Fixed C1L source quantiles for one chronological soybean year; no current labels."""
from __future__ import annotations
from _paths import method_path, method_config, earth_engine_project

import hashlib
import json
import msvcrt
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from tv5_reference import ReferenceBroker

ROOT = method_path('soybean')
QA1 = Path(str(method_path('qa1_code')))
# The long-lived monitor was started before this TV5-only cache helper existed.
# Resolve local helper imports from this file's directory, independent of the
# launcher's working directory or an earlier importlib directory snapshot.
LOCAL_CODE = Path(__file__).resolve().parent
if str(LOCAL_CODE) not in sys.path:
    sys.path.insert(0, str(LOCAL_CODE))
sys.path.insert(0, str(QA1))
from tv2_pipeline.loss import loss_pair  # noqa: E402
from tv2_pipeline.models import fit_source_pair  # noqa: E402
from tv2_pipeline.provenance import digest  # noqa: E402
from tv5_canonical_fit_cache import LockedCanonicalSourceFitCache  # noqa: E402

C1L = Path(str(method_path('corn_reference')))
KEYS = ["state", "year", "county_fips"]
QCOLS = [f"{action}_q{q}_mg_ha" for action in ("A0", "A1", "AS") for q in (10, 50, 90)]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_sha(frame: pd.DataFrame, columns: list[str]) -> str:
    part = frame[columns].copy().replace([np.inf, -np.inf], np.nan)
    row_hash = pd.util.hash_pandas_object(part, index=False).to_numpy(dtype="uint64")
    return hashlib.sha256(row_hash.tobytes()).hexdigest()


@contextmanager
def source_context_lock(path: Path):
    """Serialize one exact source context across parallel fixed-task workers."""
    with path.open("a+b") as stream:
        stream.write(b"0")
        stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


class Inputs:
    def __init__(self, year: int):
        if year not in range(2019, 2025):
            raise ValueError("Unregistered soybean target year")
        self.year = year
        freeze = json.loads((ROOT / "audit" / "TV5_PREOUTCOME_EXACT_KEY_FREEZE.json").read_text(encoding="utf-8"))
        if freeze["status"] != "PASS":
            raise RuntimeError("Soybean exact-key pre-outcome gate did not pass")
        self.freeze = freeze
        self.states = tuple(freeze["development_states"])
        self.groups = json.loads((ROOT / "TV5_SOYBEAN_FEATURE_GROUPS.json").read_text(encoding="utf-8"))
        self.rules = json.loads((C1L / "00B_TARGET_AND_MODEL_RULE_FREEZE.json").read_text(encoding="utf-8"))
        feature_path = ROOT / "TV5_SOYBEAN_PREDICTOR_PANEL.parquet"
        self.feature_sha = sha(feature_path)
        if self.feature_sha != freeze["predictor_panel_sha256"]:
            raise RuntimeError("Predictor panel SHA drift")
        f = pd.read_parquet(feature_path)
        f = f.loc[f.PRIMARY_CROP_SUPPORT & f.A1_available & f.gridmet_daily_precip_complete].copy()
        f = f.rename(columns={"target_year": "year"}).sort_values(KEYS).reset_index(drop=True)
        if any(x in f.columns for x in ("yield_mg_ha", "residual_mg_ha", "benefit", "PB3")):
            raise RuntimeError("Outcome in predictor role")
        self.features = f
        safe_history = pd.read_parquet(ROOT / "TV5_SAFE_SOYBEAN_HISTORY_2000_2018.parquet")
        reveal_path = ROOT / "chronology" / "TV5_REVEALED_PRIOR_HISTORY.parquet"
        revealed = pd.read_parquet(reveal_path) if reveal_path.exists() else safe_history.iloc[0:0]
        if not revealed.empty and (revealed.year.min() < 2019 or revealed.year.max() >= year):
            raise RuntimeError("Future/current soybean outcome in history input")
        self.raw_history = pd.concat([safe_history, revealed], ignore_index=True)
        if self.raw_history.year.max() >= year or self.raw_history.duplicated(["county_fips", "year"]).any():
            raise RuntimeError("Soybean source history chronology/identity invalid")
        self.broker = ReferenceBroker(self.raw_history, set(self.states))
        pre2019 = pd.read_parquet(ROOT / "TV5_PRE2019_HISTORY_REFERENCE_PANEL.parquet")
        if not revealed.empty:
            needed = ["state", "county_fips", "year", "yield_mg_ha", "B_history_mg_ha", "residual_mg_ha", "max_history_year"]
            if not set(needed) <= set(revealed):
                raise RuntimeError("Revealed prior-year residual schema incomplete")
            residuals = pd.concat([pre2019, revealed], ignore_index=True)
        else:
            residuals = pre2019
        self.residuals = residuals.loc[residuals.year.lt(year)].copy()
        if self.residuals.duplicated(KEYS).any():
            raise RuntimeError("Duplicate source residual key")
        if not self.residuals.max_history_year.lt(self.residuals.year).all():
            raise RuntimeError("Source residual reference chronology invalid")
        self.cache_root = ROOT / "source_cache"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.fit_cache = LockedCanonicalSourceFitCache(ROOT / "source_fit_cache")
        self.cache_access = []
        self.rules_sha = sha(C1L / "00B_TARGET_AND_MODEL_RULE_FREEZE.json")
        self.groups_sha = sha(ROOT / "TV5_SOYBEAN_FEATURE_GROUPS.json")
        self.fit_cache_identity = {
            "feature_registry_sha": self.groups_sha,
            "reference_protocol_sha": digest({"TV5_reference_code": sha(ROOT / "code" / "tv5_reference.py"),
                                               "source_rules": self.rules_sha}),
            "code_sha": digest({"QA1_models": sha(QA1 / "tv2_pipeline" / "models.py"),
                                "QA1_canonical_cache": sha(QA1 / "tv2_pipeline" / "cache.py"),
                                "TV5_source_wrapper": sha(Path(__file__)),
                                "TV5_cache_wrapper": sha(ROOT / "code" / "tv5_canonical_fit_cache.py")}),
            "source_file_shas": {"predictor_panel": self.feature_sha,
                                 "safe_pre2019_history": sha(ROOT / "TV5_SAFE_SOYBEAN_HISTORY_2000_2018.parquet"),
                                 "pre2019_reference": sha(ROOT / "TV5_PRE2019_HISTORY_REFERENCE_PANEL.parquet")},
            "preprocessing_protocol_sha": digest({"C1L_features": sha(C1L / "c1l_features.py"),
                                                  "soybean_feature_groups": self.groups_sha}),
        }

    def _feature_target(self, state: str, year: int, *, external: bool = False) -> pd.DataFrame:
        if year > self.year:
            raise RuntimeError("Future predictor target requested")
        if external:
            path = ROOT / "audit" / "TV5_EXTERNAL_EXACT_KEYS.csv"
            exact = pd.read_csv(path, dtype={"state": str, "county_fips": str})
            exact = exact.loc[exact.target_year.eq(year), ["state", "county_fips", "target_year"]].rename(columns={"target_year": "year"})
            target = self.features.merge(exact, on=KEYS, validate="one_to_one")
        else:
            target = self.features.loc[self.features.state.eq(state) & self.features.year.eq(year)].copy()
        if target.empty or target.duplicated(KEYS).any():
            raise RuntimeError(f"Empty or duplicate target predictor keys {state} {year}")
        return target.sort_values(KEYS).reset_index(drop=True)

    def prediction(self, *, allowed: tuple[str, ...], target_state: str, target_year: int,
                   geometry: str, include_as: bool = False, external: bool = False) -> pd.DataFrame:
        if not set(allowed) <= set(self.states) or target_year > self.year:
            raise RuntimeError("Unregistered source context")
        excluded = set(self.states) - set(allowed)
        if external and target_state != "EXTERNAL_ALL":
            raise RuntimeError("External context identity invalid")
        if geometry == "G1" and not external and target_state not in excluded:
            raise RuntimeError("G1 target state not excluded")
        if geometry == "G0" and target_state not in allowed:
            raise RuntimeError("G0 target state absent from source domain")
        if geometry not in ("G0", "G1"):
            raise RuntimeError("Invalid source geometry")
        target = self._feature_target(target_state, target_year, external=external)
        B = self.broker.for_keys(target[KEYS])
        target = target.merge(B[KEYS + ["B_history_mg_ha", "max_history_year"]], on=KEYS, validate="one_to_one")
        if not target.max_history_year.lt(target.year).all():
            raise RuntimeError("Target reference leakage")
        train = self.features.loc[self.features.state.isin(allowed) & self.features.year.lt(target_year)].copy()
        train = train.merge(self.residuals[KEYS + ["residual_mg_ha"]], on=KEYS, validate="one_to_one")
        if train.empty or train.year.max() >= target_year or set(train.state) & excluded:
            raise RuntimeError("Source training leakage or empty")
        train = train.sort_values(KEYS).reset_index(drop=True)
        train_cols = KEYS + ["residual_mg_ha"] + sorted(set(self.groups["A0"] + self.groups["A1"] + (self.groups["AS"] if include_as else [])))
        target_cols = KEYS + ["B_history_mg_ha", "max_history_year"] + sorted(set(self.groups["A0"] + self.groups["A1"] + (self.groups["AS"] if include_as else [])))
        spec = {"schema": "TV5.source_context.v1", "crop": "SOYBEANS", "allowed_states": list(allowed),
                "excluded_states": sorted(excluded), "target_state": target_state, "target_year": target_year,
                "geometry": geometry, "include_as": include_as, "train_content_sha256": frame_sha(train, train_cols),
                "target_content_sha256": frame_sha(target, target_cols), "feature_panel_sha256": self.feature_sha,
                "feature_groups_sha256": self.groups_sha, "source_rules_sha256": self.rules_sha,
                "QA1_corrected_models_sha256": sha(QA1 / "tv2_pipeline" / "models.py"),
                "TV5_source_model_sha256": sha(Path(__file__))}
        key = digest(spec)
        meta_path = self.cache_root / f"{key}.json"
        data_path = self.cache_root / f"{key}.parquet"
        with source_context_lock(self.cache_root / f"{key}.lock"):
            if meta_path.exists() and data_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta["spec"] != spec or meta["payload_sha256"] != sha(data_path):
                    raise RuntimeError("Soybean source cache identity/hash drift")
                cached = pd.read_parquet(data_path)
                if not cached[KEYS].equals(target[KEYS]) or len(cached) != len(target):
                    raise RuntimeError("Soybean source cache target key drift")
                self.cache_access.append({"key": key, "status": "HIT", "sha256": meta["payload_sha256"]})
                return cached
            predict = target.drop(columns=["max_history_year"])
            result, selections, fit_count = fit_source_pair(train, predict, self.groups,
                                                             label_year=target_year, excluded_states=excluded,
                                                             held_label_state=target_state if geometry == "G1" and not external else None,
                                                             source_rules=self.rules, threads=1,
                                                             source_fit_cache=self.fit_cache,
                                                             cache_identity=self.fit_cache_identity)
            out = target.copy()
            for action in ("A0", "A1"):
                full = result[action] + out.B_history_mg_ha.to_numpy(float)[:, None]
                for q, values in zip((10, 50, 90), full.T):
                    out[f"{action}_q{q}_mg_ha"] = values
            if include_as:
                pseudo = {"A0": self.groups["AS"], "A1": self.groups["AS"]}
                as_result, as_selection, extra = fit_source_pair(train, predict, pseudo,
                                                                   label_year=target_year, excluded_states=excluded,
                                                                   source_rules=self.rules, threads=1,
                                                                   source_fit_cache=self.fit_cache,
                                                                   cache_identity=self.fit_cache_identity)
                full = as_result["A0"] + out.B_history_mg_ha.to_numpy(float)[:, None]
                for q, values in zip((10, 50, 90), full.T):
                    out[f"AS_q{q}_mg_ha"] = values
                selections["AS"] = as_selection["A0"]
                fit_count += extra
            if any(x in out for x in ("yield_mg_ha", "residual_mg_ha", "L0_PB3", "L1_PB3", "delta")):
                raise RuntimeError("Outcome in frozen source prediction")
            temp_data = self.cache_root / f"{key}.{os.getpid()}.partial.parquet"
            out.to_parquet(temp_data, index=False, compression="zstd")
            payload_sha = sha(temp_data)
            os.replace(temp_data, data_path)
            temp_meta = self.cache_root / f"{key}.{os.getpid()}.partial.json"
            temp_meta.write_text(json.dumps({"spec": spec, "payload_sha256": payload_sha,
                                             "source_fits": fit_count, "source_selection": selections,
                                             "target_rows": len(out), "current_target_outcome_opened": False}, indent=2) + "\n", encoding="utf-8")
            os.replace(temp_meta, meta_path)
            self.cache_access.append({"key": key, "status": "NEW", "sha256": payload_sha, "source_fits": fit_count})
            return out

    def meta(self, *, allowed: tuple[str, ...], target_state: str, target_year: int,
             geometry: str) -> pd.DataFrame:
        if target_year >= self.year:
            raise RuntimeError("Current/future target outcome requested for meta label")
        pred = self.prediction(allowed=allowed, target_state=target_state, target_year=target_year,
                               geometry=geometry)
        outcome = self.residuals.loc[self.residuals.state.eq(target_state) & self.residuals.year.eq(target_year),
                                     KEYS + ["yield_mg_ha"]]
        frame = pred.merge(outcome, on=KEYS, how="left", validate="one_to_one")
        frame = frame.loc[frame.yield_mg_ha.notna()].copy()
        if frame.empty:
            raise RuntimeError(f"No legal source meta labels {target_state} {target_year}")
        a0 = frame[[f"A0_q{q}_mg_ha" for q in (10, 50, 90)]].to_numpy(float)
        a1 = frame[[f"A1_q{q}_mg_ha" for q in (10, 50, 90)]].to_numpy(float)
        l0, l1, delta = loss_pair(frame.yield_mg_ha.to_numpy(float), a0, a1)
        frame["L0_PB3"], frame["L1_PB3"], frame["delta"] = l0, l1, delta
        return frame
