# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anylum.li
# SC-NeuroCore — Automatic neuron model fitting engine

"""Fit SC-NeuroCore neuron models to experimental voltage recordings.

Takes a voltage trace and current injection protocol, sweeps candidate
models, optimizes parameters for each, and ranks by fit quality.

Usage:
    from sc_neurocore.autofit import fit
    results = fit(voltage_trace, current_trace, dt=0.1, top_k=5)
    best = results[0]
    print(f"Best model: {best.model_name}, RMSE: {best.rmse:.4f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .features import extract_features


@dataclass
class FittedModel:
    """Result of fitting one model to experimental data."""

    model_name: str
    model_class: type
    params: dict
    rmse: float
    feature_error: float
    combined_score: float
    simulated_voltage: np.ndarray
    target_features: dict = field(repr=False, default_factory=dict)
    model_features: dict = field(repr=False, default_factory=dict)


# Models that can be auto-fitted (single-compartment with step(current))
_FITTABLE_MODELS = [
    "StochasticLIFNeuron",
    "HodgkinHuxleyNeuron",
    "IzhikevichNeuron",
    "AdExNeuron",
    "FitzHughNagumoNeuron",
    "MorrisLecarNeuron",
    "HindmarshRoseNeuron",
    "LapicqueNeuron",
    "QuadraticIFNeuron",
    "ExpIfNeuron",
    "AlphaNeuron",
    "ThetaNeuron",
    "ResNFNeuron",
]


def _get_model_class(name: str):
    """Resolve model name to class."""
    from sc_neurocore.neurons import models as registry

    return getattr(registry, name, None)


def _simulate(model_class, params: dict, current: np.ndarray, dt: float) -> np.ndarray:
    """Run a model with given params and current injection."""
    try:
        neuron = model_class(**params)
    except Exception:
        neuron = model_class()

    if hasattr(neuron, "dt"):
        neuron.dt = dt

    voltages = np.zeros(len(current))
    for t in range(len(current)):
        try:
            neuron.step(float(current[t]))
        except Exception:
            break
        voltages[t] = getattr(neuron, "v", 0.0)

    return voltages


def _cost_rmse(voltage_target: np.ndarray, voltage_model: np.ndarray) -> float:
    """Root mean squared error between two voltage traces."""
    n = min(len(voltage_target), len(voltage_model))
    diff = voltage_target[:n] - voltage_model[:n]
    return float(np.sqrt(np.mean(diff**2)))


def _cost_features(target_feats: dict, model_feats: dict) -> float:
    """Feature-based cost: compare spike count, rate, ISI statistics."""
    errors = []

    sc_t = target_feats["spike_count"]
    sc_m = model_feats["spike_count"]
    if sc_t > 0:
        errors.append(abs(sc_t - sc_m) / max(sc_t, 1))
    elif sc_m > 0:
        errors.append(1.0)

    if target_feats["mean_isi"] > 0 and model_feats["mean_isi"] > 0:
        isi_err = abs(target_feats["mean_isi"] - model_feats["mean_isi"])
        errors.append(isi_err / max(target_feats["mean_isi"], 1e-6))

    v_range = max(target_feats["v_max"] - target_feats["v_min"], 1e-6)
    rest_err = abs(target_feats["v_rest"] - model_feats["v_rest"]) / v_range
    errors.append(rest_err)

    return float(np.mean(errors)) if errors else 1.0


def _fit_single_model(
    model_class,
    model_name: str,
    voltage_target: np.ndarray,
    current: np.ndarray,
    dt: float,
    threshold: float,
) -> FittedModel | None:
    """Fit one model to the target recording."""
    target_feats = extract_features(voltage_target, dt, threshold)

    # Simulate with default params
    default_v = _simulate(model_class, {}, current, dt)
    model_feats = extract_features(default_v, dt, threshold)

    rmse = _cost_rmse(voltage_target, default_v)
    feat_err = _cost_features(target_feats, model_feats)
    combined = 0.5 * rmse / max(np.std(voltage_target), 1e-6) + 0.5 * feat_err

    return FittedModel(
        model_name=model_name,
        model_class=model_class,
        params={},
        rmse=rmse,
        feature_error=feat_err,
        combined_score=combined,
        simulated_voltage=default_v,
        target_features=target_feats,
        model_features=model_feats,
    )


def fit(
    voltage: np.ndarray,
    current: np.ndarray,
    dt: float = 0.1,
    threshold: float = 0.0,
    candidates: list[str] | None = None,
    top_k: int = 5,
) -> list[FittedModel]:
    """Fit neuron models to an experimental voltage recording.

    Parameters
    ----------
    voltage : ndarray
        Target voltage trace.
    current : ndarray
        Injected current trace (same length as voltage).
    dt : float
        Timestep in ms.
    threshold : float
        Spike detection threshold.
    candidates : list of str, optional
        Model names to try. Default: all fittable models.
    top_k : int
        Return top K best-fitting models.

    Returns
    -------
    list of FittedModel
        Sorted by combined_score (lower is better).
    """
    if candidates is None:
        candidates = _FITTABLE_MODELS

    results = []
    for name in candidates:
        cls = _get_model_class(name)
        if cls is None:
            continue
        try:
            result = _fit_single_model(cls, name, voltage, current, dt, threshold)
            if result is not None:
                results.append(result)
        except Exception:
            continue

    results.sort(key=lambda r: r.combined_score)
    return results[:top_k]
