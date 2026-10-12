"""Frozen C1L-style chronology-safe soybean Huber reference component."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor


def huber(history: pd.DataFrame, year: int) -> tuple[float, int]:
    x = (history.year.to_numpy(float) - 2000).reshape(-1, 1)
    y = history.yield_mg_ha.to_numpy(float)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        model = HuberRegressor()
        model.fit(x, y)
    return float(model.predict(np.asarray([[year - 2000.0]]))[0]), sum(issubclass(w.category, ConvergenceWarning) for w in captured)


@dataclass
class ReferenceBroker:
    history: pd.DataFrame
    development_states: set[str]

    def __post_init__(self) -> None:
        required = {"state", "county_fips", "year", "yield_mg_ha"}
        if not required <= set(self.history) or self.history.duplicated(["county_fips", "year"]).any():
            raise ValueError("Reference history schema or key invalid")
        self.history = self.history.sort_values(["state", "county_fips", "year"]).reset_index(drop=True)
        self.by_county = {key: part for key, part in self.history.groupby("county_fips", sort=False)}
        self.by_state = {key: part for key, part in self.history.groupby("state", sort=False)}
        self.memo = {}

    def one(self, state: str, fips: str, year: int) -> dict:
        key = (fips, year)
        if key in self.memo:
            return self.memo[key]
        county = self.by_county.get(fips)
        county_history = county.loc[county.year.lt(year)] if county is not None else self.history.iloc[0:0]
        if len(county_history) >= 5 and int(county_history.year.max() - county_history.year.min()) >= 4:
            reference_history = county_history
            level = "SAME_COUNTY_HUBER"
        else:
            state_history = self.by_state.get(state)
            state_history = state_history.loc[state_history.year.lt(year)] if state_history is not None else self.history.iloc[0:0]
            if not state_history.empty:
                reference_history = state_history
                level = "SAME_STATE_HUBER"
            else:
                reference_history = self.history.loc[self.history.state.isin(self.development_states) & self.history.year.lt(year)]
                level = "SIX_DEVELOPMENT_STATE_POOLED_HUBER"
        if reference_history.empty or int(reference_history.year.max()) >= year:
            raise RuntimeError(f"No legal prior history for {fips} {year}")
        B, warning_count = huber(reference_history, year)
        result = {"state": state, "county_fips": fips, "year": year,
                  "B_history_mg_ha": B, "reference_level": level,
                  "history_rows": len(reference_history), "max_history_year": int(reference_history.year.max()),
                  "same_county_prior_rows": len(county_history), "huber_convergence_warnings": warning_count}
        self.memo[key] = result
        return result

    def for_keys(self, keys: pd.DataFrame) -> pd.DataFrame:
        if not {"state", "county_fips", "year"} <= set(keys) or keys.duplicated(["county_fips", "year"]).any():
            raise ValueError("Reference keys invalid")
        rows = [self.one(str(r.state), str(r.county_fips), int(r.year)) for r in keys.itertuples(index=False)]
        out = pd.DataFrame(rows)
        if not out.max_history_year.lt(out.year).all():
            raise RuntimeError("Reference chronology violation")
        return out
