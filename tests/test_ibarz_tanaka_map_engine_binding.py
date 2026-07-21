# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Ibarz-Tanaka map engine-binding contracts

"""Installed-extension contracts for the Ibarz-Tanaka map binding."""

from __future__ import annotations

import importlib
import sys
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import sc_neurocore_engine as engine
from sc_neurocore.neurons.models import ibarz_tanaka_map
from sc_neurocore.neurons.models.ibarz_tanaka_map import IbarzTanakaMapNeuron

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def _direct(n_steps: int) -> tuple[NDArray[np.float64], int, float, float]:
    result: tuple[Any, int, float, float] = extension.py_ibarz_tanaka_map_simulate(
        -0.5,
        -0.1,
        3.2,
        0.001,
        -1.0,
        n_steps,
        0.0,
    )
    trace, events, v_final, u_final = result
    return np.asarray(trace, dtype=np.float64), int(events), float(v_final), float(u_final)


def test_exported_name_signature_and_top_level_identity_are_stable() -> None:
    function = extension.py_ibarz_tanaka_map_simulate

    assert function.__name__ == "py_ibarz_tanaka_map_simulate"
    assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert function.__text_signature__ == "(v0, u0, alpha, mu, sigma, n_steps, current)"
    assert engine.py_ibarz_tanaka_map_simulate is function
    assert "py_ibarz_tanaka_map_simulate" in engine.__all__


def test_empty_and_initial_updates_preserve_array_and_state_contracts() -> None:
    empty_trace, empty_events, empty_v, empty_u = _direct(0)
    assert empty_trace.shape == (0,)
    assert empty_trace.dtype == np.float64
    assert empty_trace.flags.c_contiguous
    assert (empty_events, empty_v, empty_u) == (0, -0.5, -0.1)

    one_trace, one_events, one_v, one_u = _direct(1)
    np.testing.assert_array_equal(one_trace, np.array([-1.4500000000000002], dtype=np.float64))
    assert one_trace.flags.c_contiguous
    assert (one_events, one_v, one_u) == (0, -1.4500000000000002, -0.1015)

    three_trace, three_events, three_v, three_u = _direct(3)
    np.testing.assert_array_equal(
        three_trace,
        np.array([-1.4500000000000002, -4.539, -5.862050000000001], dtype=np.float64),
    )
    assert (three_events, three_v, three_u) == (0, -5.862050000000001, -0.099511)


@pytest.mark.parametrize(
    ("n_steps", "error", "message"),
    (
        (-1, OverflowError, "can't convert negative int to unsigned"),
        (1.5, TypeError, "'float' object cannot be interpreted as an integer"),
    ),
)
def test_step_count_conversion_errors_are_stable(
    n_steps: object, error: type[BaseException], message: str
) -> None:
    with pytest.raises(error) as captured:
        extension.py_ibarz_tanaka_map_simulate(-0.5, -0.1, 3.2, 0.001, -1.0, n_steps, 0.0)
    assert str(captured.value) == message
    if sys.version_info >= (3, 11):
        assert captured.value.__notes__ == ["while processing 'n_steps'"]


def test_production_rust_backend_is_exactly_the_installed_extension() -> None:
    assert ibarz_tanaka_map._HAS_RUST is True
    assert ibarz_tanaka_map._rust_simulate is engine.py_ibarz_tanaka_map_simulate

    rust_neuron = IbarzTanakaMapNeuron()
    python_neuron = IbarzTanakaMapNeuron()
    rust_trace, rust_events = rust_neuron.simulate(500, 0.2, backend="rust")
    python_trace, python_events = python_neuron.simulate(500, 0.2, backend="python")

    np.testing.assert_array_equal(rust_trace, python_trace)
    assert rust_events == python_events
    assert (rust_neuron.v, rust_neuron.u) == (python_neuron.v, python_neuron.u)
