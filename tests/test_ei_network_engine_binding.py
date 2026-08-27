# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Excitatory/inhibitory network engine-binding contracts

"""Installed-extension contracts for the seeded E/I network binding."""

from __future__ import annotations

import importlib
import sys
from typing import Any, cast

import numpy as np
import pytest

from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine
from sc_neurocore_engine.studio import get_ei_network_simulator

extension = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def _simulate() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        extension.py_simulate_ei_network(
            n_exc=20,
            n_inh=5,
            duration=20.0,
            dt=0.1,
            ext_rate=5_000.0,
            seed=7,
        ),
    )


def test_exported_name_signature_and_bridge_identity_are_stable() -> None:
    function = extension.py_simulate_ei_network

    assert function.__name__ == "py_simulate_ei_network"
    assert function.__module__ == "sc_neurocore_engine.sc_neurocore_engine"
    assert function.__text_signature__ == (
        "(n_exc=80, n_inh=20, w_ee=0.1, w_ei=0.4, w_ie=0.1, "
        "w_ii=0.4, p_conn=0.2, ext_rate=5.0, duration=200.0, dt=0.1, seed=42)"
    )
    assert engine.py_simulate_ei_network is function
    assert get_ei_network_simulator() is function


def test_seeded_spiking_trace_is_deterministic_and_typed() -> None:
    first = _simulate()
    second = _simulate()

    assert first["n_exc"] == 20
    assert first["n_inh"] == 5
    assert first["n_total"] == 25
    assert first["n_spikes"] == 48
    assert first["mean_exc_rate"] == 57.6
    assert first["mean_inh_rate"] == 250.0
    for key in ("spike_times", "spike_neurons", "rate_time", "exc_rates", "inh_rates"):
        np.testing.assert_array_equal(first[key], second[key])
        assert first[key].flags.c_contiguous
    assert first["spike_times"].dtype == np.float64
    assert first["spike_neurons"].dtype == np.int64
    np.testing.assert_array_equal(
        first["spike_neurons"][:10], [11, 15, 12, 16, 20, 13, 7, 19, 3, 4]
    )


def test_short_quiescent_network_preserves_rate_grid_and_empty_arrays() -> None:
    result = extension.py_simulate_ei_network(n_exc=2, n_inh=1, duration=1.0, dt=0.1, seed=42)

    assert result["n_total"] == 3
    assert result["n_spikes"] == 0
    assert result["spike_times"].shape == (0,)
    assert result["spike_neurons"].shape == (0,)
    np.testing.assert_array_equal(result["rate_time"], np.arange(10) * 0.1)
    np.testing.assert_array_equal(result["exc_rates"], np.zeros(10))
    np.testing.assert_array_equal(result["inh_rates"], np.zeros(10))


@pytest.mark.parametrize(
    ("n_exc", "error", "message"),
    (
        (-1, OverflowError, "can't convert negative int to unsigned"),
        (1.5, TypeError, "'float' object cannot be interpreted as an integer"),
    ),
)
def test_population_size_conversion_errors_are_stable(
    n_exc: object, error: type[BaseException], message: str
) -> None:
    with pytest.raises(error) as captured:
        extension.py_simulate_ei_network(n_exc=n_exc)
    assert str(captured.value) == message
    if sys.version_info >= (3, 11):
        assert captured.value.__notes__ == ["while processing 'n_exc'"]
