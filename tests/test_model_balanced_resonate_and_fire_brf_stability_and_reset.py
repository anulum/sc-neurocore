# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBRFStabilityAndReset from former test_model_balanced_resonate_and_fire.py

"""Focused suite: TestBRFStabilityAndReset from former test_model_balanced_resonate_and_fire.py."""

from __future__ import annotations

from tests.model_balanced_resonate_and_fire_support import *  # noqa: F403


class TestBRFStabilityAndReset:
    def test_invalid_boundary_fails_fast(self) -> None:
        with pytest.raises(ValueError, match=r"dt \* omega"):
            BalancedResonateAndFireNeuron(omega=200.0, dt=0.01)
        with pytest.raises(ValueError, match=r"dt \* omega"):
            sustain_oscillation_boundary(200.0, 0.01)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"dt": 0.0}, "dt"),
            ({"omega": 0.0}, "omega"),
            ({"b_offset": 0.0}, "b_offset"),
            ({"threshold": 0.0}, "threshold"),
            ({"gamma": 1.0}, "gamma"),
            ({"x": float("nan")}, "finite"),
        ],
    )
    def test_parameter_validation(self, kwargs: dict[str, float], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            BalancedResonateAndFireNeuron(**kwargs)

    def test_refractory_period_raises_threshold_and_decays(self) -> None:
        neuron = BalancedResonateAndFireNeuron()
        assert neuron.step(200.0) == 1
        assert neuron.q == pytest.approx(1.0)
        assert neuron.dynamic_threshold == pytest.approx(2.0)

        neuron.step(0.0)
        assert 0.0 < neuron.q < 1.0
        assert neuron.dynamic_threshold == pytest.approx(1.9)

    def test_smooth_reset_preserves_phase_state_after_spike(self) -> None:
        neuron = BalancedResonateAndFireNeuron()
        assert neuron.step(200.0) == 1
        assert neuron.x != 0.0
        assert neuron.y == 0.0
        assert neuron.damping < neuron.p_omega - neuron.b_offset

    def test_state_remains_finite_under_long_drive(self) -> None:
        neuron = BalancedResonateAndFireNeuron(omega=20.0, b_offset=2.0)
        for _ in range(20_000):
            neuron.step(5.0)
        snapshot = neuron.state()
        assert all(math.isfinite(value) for value in snapshot.values())

    def test_reset_clears_membrane_and_refractory_state(self) -> None:
        neuron = BalancedResonateAndFireNeuron()
        neuron.step(200.0)
        neuron.reset()
        assert neuron.x == 0.0
        assert neuron.y == 0.0
        assert neuron.q == 0.0
