# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Module-level tests from former test_model_booth_rinzel.py

"""Module-level tests from former test_model_booth_rinzel.py."""

from __future__ import annotations

from tests.model_booth_rinzel_support import *  # noqa: F403

@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dt", 0.0),
        ("p", 0.0),
        ("p", 1.0),
        ("gc", 0.0),
        ("g_na", 0.0),
        ("g_k", 0.0),
        ("g_ca", 0.0),
        ("g_kca", 0.0),
        ("g_l", 0.0),
        ("c_m", 0.0),
        ("alpha_ca", 0.0),
        ("k_ca", 0.0),
        ("f_ca", 0.0),
        ("h", -0.01),
        ("n", 1.01),
        ("q", float("nan")),
        ("ca", -0.01),
    ],
)
def test_booth_rinzel_rejects_invalid_physical_configuration(field, value):
    kwargs = {field: value}
    with pytest.raises(ValueError):
        BoothRinzelNeuron(**kwargs)
def test_booth_rinzel_runtime_validation_is_fail_closed():
    neuron = BoothRinzelNeuron()
    neuron.p = 1.0
    before = _booth_state_tuple(neuron)
    with pytest.raises(ValueError):
        neuron.step(10.0)
    assert _booth_state_tuple(neuron) == before
def test_booth_rinzel_nonfinite_input_is_fail_closed():
    neuron = BoothRinzelNeuron()
    before = _booth_state_tuple(neuron)
    with pytest.raises(ValueError):
        neuron.step(float("nan"))
    assert _booth_state_tuple(neuron) == before
def test_booth_rinzel_drive_preserves_physical_bounds():
    neuron = BoothRinzelNeuron()
    for _ in range(100):
        neuron.step(8.0)
        assert -200.0 <= neuron.vs <= 100.0
        assert -200.0 <= neuron.vd <= 100.0
        assert 0.0 <= neuron.h <= 1.0
        assert 0.0 <= neuron.n <= 1.0
        assert 0.0 <= neuron.q <= 1.0
        assert neuron.ca >= 0.0
