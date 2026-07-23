# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Module-level tests from former test_model_butera_respiratory.py

"""Module-level tests from former test_model_butera_respiratory.py."""

from __future__ import annotations

from tests.model_butera_respiratory_support import *  # noqa: F403

def test_butera_matches_independent_rk4_contract() -> None:
    """Butera respiratory step follows the module RK4 integration contract."""
    neuron = ButeraRespiratoryNeuron(v=-48.0, n=0.08, h_nap=0.62, dt=0.025)
    expected = _butera_reference_rk4(neuron, current=18.0)

    spike = neuron.step(18.0)

    assert spike in (0, 1)
    assert (neuron.v, neuron.n, neuron.h_nap) == pytest.approx(expected, rel=1e-10, abs=1e-10)
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("v", float("nan")),
        ("n", -0.01),
        ("h_nap", 1.01),
        ("g_na", -1.0),
        ("g_nap", -1.0),
        ("g_k", -1.0),
        ("g_l", -1.0),
        ("tau_h", 0.0),
        ("dt", 0.0),
    ],
)
def test_butera_rejects_invalid_physical_parameters(field: str, value: float) -> None:
    """Invalid Butera state or physical parameters are rejected at construction."""
    with pytest.raises((TypeError, ValueError)):
        ButeraRespiratoryNeuron(**{field: value})
def test_butera_rejects_non_finite_current_without_mutation() -> None:
    """Invalid respiratory drive preserves voltage and gate state."""
    neuron = ButeraRespiratoryNeuron(v=-49.0, n=0.04, h_nap=0.55)
    before = (neuron.v, neuron.n, neuron.h_nap)

    with pytest.raises((TypeError, ValueError, FloatingPointError)):
        neuron.step(float("nan"))

    assert (neuron.v, neuron.n, neuron.h_nap) == before
def test_butera_rejects_corrupted_runtime_state_without_mutation() -> None:
    """Runtime gate corruption cannot produce a partially committed candidate."""
    neuron = ButeraRespiratoryNeuron()
    neuron.n = float("inf")
    before = (neuron.v, neuron.n, neuron.h_nap)

    with pytest.raises((TypeError, ValueError, FloatingPointError)):
        neuron.step(20.0)

    assert neuron.v == before[0]
    assert np.isinf(neuron.n)
    assert neuron.h_nap == before[2]
