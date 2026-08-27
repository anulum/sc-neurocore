# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wilson-HR engine-binding contracts

"""Installed-extension contracts for the Wilson-HR binding."""

from __future__ import annotations

import importlib
import sys
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine
from sc_neurocore.neurons.models import wilson_hr
from sc_neurocore.neurons.models.wilson_hr import WilsonHRNeuron

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def _direct(n_steps: int) -> tuple[NDArray[np.float64], int, float, float]:
    result: tuple[Any, int, float, float] = extension.py_wilson_hr_simulate(
        -0.7, 0.1, 1.9, 0.4, 0.05, n_steps, 2.0
    )
    trace, spikes, final_v, final_r = result
    return np.asarray(trace, dtype=np.float64), int(spikes), float(final_v), float(final_r)


def test_exported_name_signature_and_top_level_identity_are_stable() -> None:
    function = extension.py_wilson_hr_simulate

    assert function.__name__ == "py_wilson_hr_simulate"
    assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert function.__text_signature__ == "(v0, r0, tau_r, v_peak, dt, n_steps, current)"
    assert engine.py_wilson_hr_simulate is function
    assert "py_wilson_hr_simulate" in engine.__all__


def test_empty_and_initial_updates_preserve_array_and_state_contracts() -> None:
    empty_trace, empty_spikes, empty_v, empty_r = _direct(0)
    assert empty_trace.shape == (0,)
    assert empty_trace.dtype == np.float64
    assert empty_trace.flags.c_contiguous
    assert (empty_spikes, empty_v, empty_r) == (0, -0.7, 0.1)

    one_trace, one_spikes, one_v, one_r = _direct(1)
    np.testing.assert_array_equal(one_trace, np.array([-0.5988676025214146], dtype=np.float64))
    assert one_trace.flags.c_contiguous
    assert (one_spikes, one_v, one_r) == (0, -0.5988676025214146, 0.10134793845659071)

    three_trace, three_spikes, three_v, three_r = _direct(3)
    np.testing.assert_array_equal(
        three_trace,
        np.array(
            [-0.5988676025214146, -0.46100801824819004, -0.21457383566794985],
            dtype=np.float64,
        ),
    )
    assert (three_spikes, three_v, three_r) == (
        0,
        -0.21457383566794985,
        0.11838433542799504,
    )


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
        extension.py_wilson_hr_simulate(-0.7, 0.1, 1.9, 0.4, 0.05, n_steps, 2.0)
    assert str(captured.value) == message
    if sys.version_info >= (3, 11):
        assert captured.value.__notes__ == ["while processing 'n_steps'"]


def test_production_rust_backend_is_exactly_the_installed_extension() -> None:
    assert wilson_hr._HAS_RUST is True
    assert wilson_hr._rust_simulate is engine.py_wilson_hr_simulate

    rust_neuron = WilsonHRNeuron()
    python_neuron = WilsonHRNeuron()
    rust_trace, rust_spikes = rust_neuron.simulate(500, 10.0, backend="rust")
    python_trace, python_spikes = python_neuron.simulate(500, 10.0, backend="python")

    np.testing.assert_array_equal(rust_trace, python_trace)
    assert rust_spikes == python_spikes
    assert (rust_neuron.v, rust_neuron.r) == (python_neuron.v, python_neuron.r)
