# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rust-Python neuron parity: parametrized over all 108

"""
Rust-Python neuron parity: parametrized over all 108 model classes.

For each model, runs identical default params through Python dataclass and
Rust engine, verifies spike counts match within tolerance.
"""

from __future__ import annotations

import pytest

from sc_neurocore.neurons import models as py_models

try:
    from sc_neurocore_engine import sc_neurocore_engine as eng

    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

pytestmark = pytest.mark.skipif(not HAS_ENGINE, reason="Rust engine not built")

# Models with non-standard step() signatures
_DUAL_FLOAT = {
    "AlphaNeuron",
    "COBALIFNeuron",
    "PinskyRinzelNeuron",
    "HayL5PyramidalNeuron",
    "TwoCompartmentLIFNeuron",
}
_BOOL_PARAM = {"TsodyksMarkramNeuron", "CompteWMNeuron"}
_INT_INPUT = {
    "LoihiCUBANeuron",
    "Loihi2Neuron",
    "TrueNorthNeuron",
    "SpiNNaker2Neuron",
    "IntegerQIFNeuron",
    "AkidaNeuron",
}
_VEC_INPUT = {"AmariNeuralField", "LeakyCompeteFireNeuron"}
_RATE_OVERRIDE = {"PoissonNeuron", "InhomogeneousPoissonNeuron", "GammaRenewalNeuron"}
# Stochastic models (RNG-dependent, skip exact parity)
_STOCHASTIC = {
    "EscapeRateNeuron",
    "BendaHerzNeuron",
    "PoissonNeuron",
    "InhomogeneousPoissonNeuron",
    "GammaRenewalNeuron",
    "StochasticIFNeuron",
    "GalvesLocherbachNeuron",
    "GLMNeuron",
    "GIFPopulationNeuron",
    "WongWangUnit",
}
# The Python ChayKeizer model is the faithful five-dimensional Chay-Keizer 1983
# burster; the Rust kernel still implements the earlier reduced three-variable
# form, so the two legitimately diverge until the Rust kernel is updated to match.
_KNOWN_PARITY_DIVERGENCE: set[str] = {"ChayKeizerNeuron"}


def _get_all_model_names():
    return [name for name in py_models.__all__]


def _make_py(name):
    return getattr(py_models, name)()


_RUST_NAME_MAP = {
    "ContinuousAttractorNeuron": "RustContinuousAttractorNeuron",
}


def _make_rs(name):
    rs_name = _RUST_NAME_MAP.get(name, name)
    if not hasattr(eng, rs_name):
        pytest.skip(f"No Rust binding for {name}")
    return getattr(eng, rs_name)()


def _run_steps(model, name, n=500):
    spikes = []
    for _ in range(n):
        if name in _VEC_INPUT:
            s = model.step([5.0] * 4 if name == "LeakyCompeteFireNeuron" else [0.5] * 64)
            spikes.append(sum(s) if isinstance(s, (list, tuple)) else int(s))
        elif name in _INT_INPUT:
            spikes.append(int(model.step(50)))
        elif name in _DUAL_FLOAT:
            spikes.append(int(model.step(5.0, 0.0)))
        elif name in _BOOL_PARAM:
            spikes.append(int(model.step(5.0, False)))
        elif name in _RATE_OVERRIDE:
            s = model.step(-1.0) if name != "InhomogeneousPoissonNeuron" else model.step(200.0)
            spikes.append(int(s) if isinstance(s, (int, bool)) else (1 if s > 0.5 else 0))
        else:
            result = model.step(5.0)
            if isinstance(result, float):
                spikes.append(1 if abs(result) > 0.001 else 0)
            elif isinstance(result, tuple):
                spikes.append(1 if any(abs(x) > 0.001 for x in result) else 0)
            else:
                spikes.append(int(result))
    return spikes


@pytest.mark.parametrize("name", _get_all_model_names())
def test_parity(name):
    if name in _STOCHASTIC:
        pytest.skip(f"{name} is RNG-dependent, skip exact parity")
    if name in _KNOWN_PARITY_DIVERGENCE:
        pytest.xfail(f"{name}: known Rust/Python parity divergence")

    py_model = _make_py(name)
    rs_model = _make_rs(name)

    n = 2000 if "Butera" in name or "Bertram" in name else 500
    py_spikes = _run_steps(py_model, name, n)
    rs_spikes = _run_steps(rs_model, name, n)

    py_count = sum(py_spikes)
    rs_count = sum(rs_spikes)

    if py_count == 0 and rs_count == 0:
        return

    max_delta = max(3, int(max(py_count, rs_count) * 0.15))
    assert abs(py_count - rs_count) <= max_delta, (
        f"{name}: Python={py_count}, Rust={rs_count}, delta={abs(py_count - rs_count)}, max_delta={max_delta}"
    )


@pytest.mark.parametrize(
    "name", ["LoihiCUBANeuron", "TrueNorthNeuron", "SigmaDeltaNeuron", "McCullochPittsNeuron"]
)
def test_exact_parity(name):
    """Integer/deterministic models must match exactly."""
    py_model = _make_py(name)
    rs_model = _make_rs(name)

    py_spikes = _run_steps(py_model, name, 200)
    rs_spikes = _run_steps(rs_model, name, 200)

    assert py_spikes == rs_spikes, f"{name}: spike trains differ"
