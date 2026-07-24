# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (rust_and_reset) from former test_model_dpi_neuron.py

from __future__ import annotations

from tests.model_dpi_neuron_support import *  # noqa: F403


def test_rust_compatibility_boundary_is_exact_factory_contract() -> None:
    """Use the fixed-constructor PyO3 engine only for an exact field match."""
    neuron = DPINeuron()
    assert neuron._matches_rust_engine_contract()
    neuron.i_ahp = math.nextafter(neuron.i_ahp, math.inf)
    assert not neuron._matches_rust_engine_contract()


def test_reset_restores_current_baselines_and_preserves_parameters() -> None:
    """Reset three dynamic states without destroying circuit configuration."""
    neuron = _configured()
    parameters = tuple(
        neuron.__dict__[name]
        for name in neuron.__dict__
        if name
        not in {
            "i_mem",
            "i_ahp",
            "refractory_time",
        }
    )
    neuron.simulate(100, 5.0, backend="python")
    neuron.reset()
    assert (neuron.i_mem, neuron.i_ahp, neuron.refractory_time) == (
        neuron.i_reset,
        neuron.i_0,
        0.0,
    )
    assert (
        tuple(
            neuron.__dict__[name]
            for name in neuron.__dict__
            if name
            not in {
                "i_mem",
                "i_ahp",
                "refractory_time",
            }
        )
        == parameters
    )
