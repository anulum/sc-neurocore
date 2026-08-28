# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Medvedev validation and reset contracts

"""Parameter, request, failure, and reset tests for the Medvedev map."""

from __future__ import annotations

import sys

import pytest

from sc_neurocore.neurons.models.medvedev_map import MedvedevMapNeuron


@pytest.mark.parametrize(
    "overrides",
    (
        {"beta_sn": 0.001},
        {"beta_hc": 0.02},
        {"decay_t0": 1.0},
        {"alpha_t0": 0.0},
        {"f_1": 2.0},
        {"homoclinic_exponent": 0.0},
        {"d": 0.0},
        {"input_gain": -1.0},
    ),
)
def test_invalid_parameter_topology_is_rejected(overrides: dict[str, float]) -> None:
    """Invalid source topology cannot enter the runtime."""
    with pytest.raises(ValueError):
        MedvedevMapNeuron(**overrides)


def test_failed_step_preserves_state() -> None:
    """Non-finite input fails before state mutation."""
    neuron = MedvedevMapNeuron()
    before = neuron.u
    with pytest.raises(ValueError, match="current must be finite"):
        neuron.step(float("nan"))
    assert neuron.u == before


def test_failed_batch_preserves_state() -> None:
    """A mutable parameter fault rejects the batch without state mutation."""
    neuron = MedvedevMapNeuron()
    before = neuron.u
    neuron.d = float("inf")
    with pytest.raises(ValueError, match="parameters must be finite"):
        neuron.simulate(10, 2.0, backend="python")
    assert neuron.u == before


def test_non_finite_state_is_rejected_at_construction_and_runtime() -> None:
    """Construction reports bad input while a corrupted live state fails closed."""
    with pytest.raises(ValueError, match="state must be finite"):
        MedvedevMapNeuron(u=float("nan"))

    neuron = MedvedevMapNeuron()
    neuron.u = float("inf")
    with pytest.raises(FloatingPointError, match="state must be finite"):
        neuron.step()


def test_non_finite_branch_arithmetic_preserves_state() -> None:
    """Finite-but-extreme inputs cannot commit invalid branch arithmetic."""
    neuron = MedvedevMapNeuron(
        u=0.5,
        beta_0=1.0,
        beta_sn=2.0,
        beta_hc=3.0,
        delta=4.0,
        d=sys.float_info.max,
    )
    before = neuron.u
    with pytest.raises(FloatingPointError, match="log argument is invalid"):
        neuron.step()
    assert neuron.u == before

    neuron = MedvedevMapNeuron(u=0.0, input_gain=sys.float_info.max)
    before = neuron.u
    with pytest.raises(FloatingPointError, match="candidate became non-finite"):
        neuron.step(sys.float_info.max)
    assert neuron.u == before


def test_request_validation() -> None:
    """Batch bounds and backend selection are explicit."""
    neuron = MedvedevMapNeuron()
    with pytest.raises(ValueError, match="n_steps must be an integer"):
        neuron.simulate(True)
    with pytest.raises(ValueError, match="n_steps must be between"):
        neuron.simulate(-1)
    with pytest.raises(ValueError, match="backend must be"):
        neuron.simulate(1, backend="cuda")


def test_reset_restores_only_the_derived_return_state() -> None:
    """Reset preserves calibration and recomputes u_SN from mutable parameters."""
    neuron = MedvedevMapNeuron()
    neuron.beta_sn = 0.0019
    neuron.u = 0.3
    neuron.reset()
    assert neuron.u == neuron.beta_sn / (neuron.delta - neuron.beta_sn)
    assert neuron.beta_sn == 0.0019
