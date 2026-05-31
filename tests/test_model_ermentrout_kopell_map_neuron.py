# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Module-specific test: ErmentroutKopellMapNeuron

from __future__ import annotations

import math

import numpy as np
import pytest

from sc_neurocore.neurons.models.ermentrout_kopell_map_neuron import (
    ErmentroutKopellMapNeuron,
)


class TestErmentroutKopellMapNeuron:
    def test_defaults_and_binary_step(self):
        n = ErmentroutKopellMapNeuron()
        assert n.theta == 0.0
        assert n.theta_threshold == math.pi
        assert n.step(0.0) in (0, 1)

    def test_positive_current_advances_phase_on_circle(self):
        n = ErmentroutKopellMapNeuron()
        n.step(1.0)
        assert 0.0 <= n.theta < 2.0 * math.pi
        assert n.theta > 0.0

    def test_phase_wrap_uses_circle_geometry(self):
        n = ErmentroutKopellMapNeuron(theta=2.0 * math.pi - 0.01, dt=1.0)
        n.step(1.0)
        assert 0.0 <= n.theta < 2.0 * math.pi

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("theta", np.nan),
            ("dt", 0.0),
            ("gain", np.inf),
            ("theta_threshold", np.nan),
        ],
    )
    def test_rejects_invalid_numerical_configuration(self, field: str, value: float):
        with pytest.raises(ValueError):
            ErmentroutKopellMapNeuron(**{field: value})

    def test_rejects_non_finite_current_before_state_mutation(self):
        n = ErmentroutKopellMapNeuron()
        before = n.theta
        with pytest.raises(ValueError, match="current"):
            n.step(np.nan)
        assert n.theta == before

    def test_rejects_corrupted_runtime_state_before_mutation(self):
        n = ErmentroutKopellMapNeuron()
        n.theta = np.inf
        before = n.theta
        with pytest.raises(FloatingPointError, match="phase state"):
            n.step(1.0)
        assert n.theta == before

    def test_rejects_non_finite_candidate_before_state_mutation(self):
        n = ErmentroutKopellMapNeuron(gain=1.0e308)
        before = n.theta
        with pytest.raises(FloatingPointError, match="input drive"):
            n.step(1.0e308)
        assert n.theta == before

    def test_spike_detects_upward_threshold_crossing(self):
        n = ErmentroutKopellMapNeuron(theta=math.pi - 0.01, dt=1.0)
        assert n.step(1.0) == 1
