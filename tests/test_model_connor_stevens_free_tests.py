# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Module-level tests from former test_model_connor_stevens.py

"""Module-level tests from former test_model_connor_stevens.py."""

from __future__ import annotations

from tests.model_connor_stevens_support import *  # noqa: F403


def test_connor_stevens_matches_independent_rk4_contract() -> None:
    """Connor-Stevens step follows the module RK4 integration contract."""
    neuron = ConnorStevensNeuron(v=-62.0, m=0.05, h=0.84, n=0.22, a=0.41, b=0.27, dt=0.02)
    expected = _connor_reference_rk4(neuron, current=8.5)

    spike = neuron.step(8.5)

    assert spike in (0, 1)
    assert (neuron.v, neuron.m, neuron.h, neuron.n, neuron.a, neuron.b) == pytest.approx(
        expected, rel=1e-10, abs=1e-10
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("v", np.nan),
        ("m", np.inf),
        ("g_na", -1.0),
        ("g_k", -1.0),
        ("g_a", -1.0),
        ("g_l", -1.0),
        ("c_m", 0.0),
        ("dt", 0.0),
    ],
)
def test_connor_stevens_rejects_invalid_parameters(field: str, value: float) -> None:
    """Invalid physical parameters are rejected before simulation begins."""
    with pytest.raises((TypeError, ValueError)):
        ConnorStevensNeuron(**{field: value})


def test_connor_stevens_rejects_non_finite_current_without_mutation() -> None:
    """Adapter-visible invalid drive fails closed and preserves biological state."""
    neuron = ConnorStevensNeuron(v=-63.0, m=0.04, h=0.91, n=0.18, a=0.36, b=0.31)
    before = (neuron.v, neuron.m, neuron.h, neuron.n, neuron.a, neuron.b)

    with pytest.raises((TypeError, ValueError, FloatingPointError)):
        neuron.step(float("nan"))

    assert (neuron.v, neuron.m, neuron.h, neuron.n, neuron.a, neuron.b) == before


def test_connor_stevens_rejects_corrupted_runtime_state_without_mutation() -> None:
    """Runtime state corruption cannot be amplified into a partially committed step."""
    neuron = ConnorStevensNeuron()
    neuron.b = float("inf")
    before = (neuron.v, neuron.m, neuron.h, neuron.n, neuron.a, neuron.b)

    with pytest.raises((TypeError, ValueError, FloatingPointError)):
        neuron.step(6.0)

    assert (neuron.v, neuron.m, neuron.h, neuron.n, neuron.a, neuron.b) == before
