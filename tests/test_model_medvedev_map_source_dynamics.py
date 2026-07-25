# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Medvedev source calibration and dynamics

"""Source calibration and branch dynamics tests for the Medvedev map."""

from __future__ import annotations

import inspect

import pytest

from sc_neurocore.neurons.model_catalogue import load_descriptor_payload
from sc_neurocore.neurons.models.medvedev_map import MedvedevMapNeuron

from .model_medvedev_map_support import _boundaries, _inner_reference


def test_defaults_are_the_disclosed_source_calibration() -> None:
    """The initial state is the Eq. 4.15 saddle-node return, not zero."""
    neuron = MedvedevMapNeuron()
    u_0, u_hc, u_sn = _boundaries(neuron)
    assert u_0 == pytest.approx(0.1764705882352941)
    assert u_hc == pytest.approx(0.25470514429109165)
    assert u_sn == pytest.approx(0.2514078836724436)
    assert neuron.u == u_sn
    assert neuron.d > 127.996  # Signed Q8.8 cannot encode the calibrated scale.


def test_descriptor_structure_matches_map_runtime() -> None:
    """The unit iteration belongs only to integration, never to parameters."""
    payload = load_descriptor_payload("MedvedevMapNeuron")
    assert payload is not None
    assert "dt" not in inspect.signature(MedvedevMapNeuron).parameters
    assert "dt" not in payload["parameters"]
    assert payload["integration"] == {"dt": 1.0, "method": "map"}
    assert set(payload["state"]) == {"u"}
    assert set(payload["parameters"]) == {
        "beta_0",
        "beta_hc",
        "beta_sn",
        "delta",
        "decay_t0",
        "alpha_t0",
        "f_0",
        "f_1",
        "homoclinic_exponent",
        "d",
        "input_gain",
    }


def test_left_branch_matches_eq_4_4_calibration() -> None:
    """The active left branch uses the calibrated exponential relaxation."""
    neuron = MedvedevMapNeuron(u=0.1)
    current = 2.0
    expected = (
        neuron.decay_t0 * neuron.u
        + (1.0 - neuron.decay_t0) * neuron.f_0
        + neuron.input_gain * current
    )
    assert neuron.step(current) == 1
    assert neuron.u == expected


def test_inner_branch_matches_eq_4_8_and_eq_4_13_calibration() -> None:
    """The middle branch composes the affine and homoclinic returns."""
    neuron = MedvedevMapNeuron(u=0.2)
    expected = _inner_reference(neuron, neuron.u, 2.0)
    assert neuron.step(2.0) == 1
    assert neuron.u == expected


def test_right_branch_is_exact_eq_4_15_return_without_input() -> None:
    """External current does not perturb the slow right return."""
    neuron = MedvedevMapNeuron(u=0.3)
    _u_0, _u_hc, u_sn = _boundaries(neuron)
    assert neuron.step(1000.0) == 0
    assert neuron.u == u_sn


def test_event_uses_pre_state_fast_return_region() -> None:
    """The event is an observation of the pre-step active branch."""
    neuron = MedvedevMapNeuron(u=0.3)
    assert neuron.step(0.0) == 0
    assert neuron.step(0.0) == 1
