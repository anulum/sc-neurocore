# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia RK4 neuron integrator parity tests

from __future__ import annotations

from typing import Protocol

import numpy as np
import pytest

from sc_neurocore.neurons.models.adex import AdExNeuron
from sc_neurocore.neurons.models.hodgkin_huxley import HodgkinHuxleyNeuron
from sc_neurocore.neurons.sc_izhikevich import SCIzhikevichNeuron

from sc_neurocore.accel.julia.neurons import simulate_rk4_neuron

from tests.julia_requirement import require_julia

require_julia()


class _ReferenceFn(Protocol):
    def __call__(self, currents: np.ndarray, *, dt: float) -> dict[str, np.ndarray]: ...


def _python_izhikevich(currents: np.ndarray, *, dt: float) -> dict[str, np.ndarray]:
    neuron = SCIzhikevichNeuron(dt=dt, noise_std=0.0, integrator="rk4")
    v: list[float] = []
    u: list[float] = []
    spikes: list[int] = []
    for idx, current in enumerate(currents):
        if neuron.step(float(current)):
            spikes.append(idx)
        state = neuron.get_state()
        v.append(state["v"])
        u.append(state["u"])
    return {
        "v": np.asarray(v, dtype=np.float64),
        "u": np.asarray(u, dtype=np.float64),
        "spikes": np.asarray(spikes, dtype=np.uint64),
    }


def _python_hodgkin_huxley(currents: np.ndarray, *, dt: float) -> dict[str, np.ndarray]:
    neuron = HodgkinHuxleyNeuron(dt=dt, integrator="rk4")
    v: list[float] = []
    m: list[float] = []
    h: list[float] = []
    n: list[float] = []
    spikes: list[int] = []
    for idx, current in enumerate(currents):
        if neuron.step(float(current)):
            spikes.append(idx)
        v.append(neuron.v)
        m.append(neuron.m)
        h.append(neuron.h)
        n.append(neuron.n)
    return {
        "v": np.asarray(v, dtype=np.float64),
        "m": np.asarray(m, dtype=np.float64),
        "h": np.asarray(h, dtype=np.float64),
        "n": np.asarray(n, dtype=np.float64),
        "spikes": np.asarray(spikes, dtype=np.uint64),
    }


def _python_adex(currents: np.ndarray, *, dt: float) -> dict[str, np.ndarray]:
    neuron = AdExNeuron(dt=dt, integrator="rk4")
    v: list[float] = []
    w: list[float] = []
    spikes: list[int] = []
    for idx, current in enumerate(currents):
        if neuron.step(float(current)):
            spikes.append(idx)
        v.append(neuron.v)
        w.append(neuron.w)
    return {
        "v": np.asarray(v, dtype=np.float64),
        "w": np.asarray(w, dtype=np.float64),
        "spikes": np.asarray(spikes, dtype=np.uint64),
    }


@pytest.mark.parametrize(
    ("model", "dt", "currents", "reference_fn", "state_keys"),
    [
        (
            "SCIzhikevichNeuron",
            1.0,
            np.full(80, 10.0, dtype=np.float64),
            _python_izhikevich,
            ("v", "u"),
        ),
        (
            "HodgkinHuxleyNeuron",
            0.02,
            np.full(60, 10.0, dtype=np.float64),
            _python_hodgkin_huxley,
            ("v", "m", "h", "n"),
        ),
        (
            "AdExNeuron",
            0.2,
            np.full(250, 500.0, dtype=np.float64),
            _python_adex,
            ("v", "w"),
        ),
    ],
)
def test_julia_rk4_neuron_simulator_matches_python(
    model: str,
    dt: float,
    currents: np.ndarray,
    reference_fn: _ReferenceFn,
    state_keys: tuple[str, ...],
) -> None:
    expected = reference_fn(currents, dt=dt)
    actual = simulate_rk4_neuron(model, currents, dt)

    assert actual["n_steps"] == len(currents)
    np.testing.assert_array_equal(np.asarray(actual["spikes"], dtype=np.uint64), expected["spikes"])
    for key in state_keys:
        np.testing.assert_allclose(np.asarray(actual[key]), expected[key], rtol=1e-10, atol=1e-10)


def test_julia_rk4_neuron_simulator_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="unsupported RK4 neuron model"):
        simulate_rk4_neuron("unknown", np.ones(4, dtype=np.float64))


@pytest.mark.parametrize(
    ("currents", "dt", "match"),
    [
        (np.array([], dtype=np.float64), 1.0, "non-empty"),
        (np.array([np.nan], dtype=np.float64), 1.0, "finite"),
        (np.ones((1, 2), dtype=np.float64), 1.0, "1-D"),
        (np.ones(4, dtype=np.float64), 0.0, "dt"),
        (np.ones(4, dtype=np.float64), np.nan, "dt"),
    ],
)
def test_julia_rk4_neuron_simulator_rejects_invalid_trace_or_dt(
    currents: np.ndarray, dt: float, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        simulate_rk4_neuron("izhikevich", currents, dt)


def test_julia_rk4_neuron_simulator_matches_rust_when_available() -> None:
    rust = pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built")
    currents = np.linspace(0.0, 12.0, 96, dtype=np.float64)

    rust_out = rust.py_rk4_neuron_simulate("izhikevich", currents, 1.0)
    julia_out = simulate_rk4_neuron("izhikevich", currents, 1.0)

    np.testing.assert_array_equal(
        np.asarray(julia_out["spikes"], dtype=np.uint64),
        np.asarray(rust_out["spikes"], dtype=np.uint64),
    )
    np.testing.assert_allclose(
        np.asarray(julia_out["v"]), np.asarray(rust_out["v"]), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        np.asarray(julia_out["u"]), np.asarray(rust_out["u"]), rtol=0, atol=0
    )
