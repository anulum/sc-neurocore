# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (rust_and_reset) from former test_model_dpi_neuron.py

from __future__ import annotations

from tests.model_dpi_neuron_support import *  # noqa: F403


def test_rust_complete_batch_accepts_configured_contract() -> None:
    """Carry non-default state and parameters through the production Rust API."""
    neuron = _configured()
    i_mem, i_ahp, refractory, events = neuron.simulate_complete(400, 5.0, backend="rust")
    assert int(np.sum(events, dtype=np.int64)) == 4
    assert (neuron.i_mem, neuron.i_ahp, neuron.refractory_time) == (
        i_mem[-1],
        i_ahp[-1],
        refractory[-1],
    )


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
