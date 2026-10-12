"""Registered PB3, action, bridge and block aggregation mathematics."""
from __future__ import annotations

import numpy as np
import pandas as pd

QUANTILES = np.asarray((.1, .5, .9))
LAMBDAS = (0., .25, .5, .75, 1.)


def pinball(y: np.ndarray, qpred: np.ndarray) -> np.ndarray:
    y, qpred = np.asarray(y, float), np.asarray(qpred, float)
    if qpred.ndim != 2 or qpred.shape[1] != 3 or len(y) != len(qpred):
        raise ValueError("PB3_SHAPE")
    e = y[:, None] - qpred
    return np.mean(np.maximum(QUANTILES * e, (QUANTILES - 1) * e), axis=1)


def loss_pair(y: np.ndarray, a0: np.ndarray, a1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    l0, l1 = pinball(y, a0), pinball(y, a1)
    return l0, l1, l0 - l1


def action_from_delta(delta_hat: np.ndarray) -> np.ndarray:
    return (np.asarray(delta_hat) > 0).astype(np.int8)  # exact tie is A0


def selected_prediction(a0: np.ndarray, a1: np.ndarray, action: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(action)[:, None] == 1, a1, a0)


def blend_prediction(a0: np.ndarray, a1: np.ndarray, lam: float) -> np.ndarray:
    if lam not in LAMBDAS:
        raise ValueError("UNREGISTERED_BLEND")
    return (1 - lam) * np.asarray(a0) + lam * np.asarray(a1)


def block_weights(state: np.ndarray, year: np.ndarray) -> np.ndarray:
    keys = pd.Series(list(zip(state, year)))
    count = keys.map(keys.value_counts()).to_numpy(float)
    weights = 1 / count
    return weights / weights.mean()


def equal_block_metric(loss: np.ndarray, state: np.ndarray, year: np.ndarray) -> float:
    frame = pd.DataFrame({"state": state, "year": year, "loss": loss})
    return float(frame.groupby(["state", "year"], sort=True).loss.mean().mean())


def source_selected_static(delta: np.ndarray, state: np.ndarray, year: np.ndarray) -> int:
    advantage = equal_block_metric(delta, state, year)
    return int(advantage > 0)


def source_selected_blend(y: np.ndarray, a0: np.ndarray, a1: np.ndarray, state: np.ndarray, year: np.ndarray) -> float:
    values = [(equal_block_metric(pinball(y, blend_prediction(a0, a1, lam)), state, year), lam) for lam in LAMBDAS]
    return min(values)[1]


def aggregate_metrics(y: np.ndarray, state: np.ndarray, year: np.ndarray, predictions: dict[str, np.ndarray]) -> dict:
    return {name: {"equal_state_year_PB3": equal_block_metric(pinball(y, pred), state, year), "n": len(y)} for name, pred in predictions.items()}
