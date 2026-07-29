# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAmariParameters from former test_model_amari_field.py

"""Focused suite: TestAmariParameters from former test_model_amari_field.py."""

from __future__ import annotations

from tests.model_amari_field_support import *  # noqa: F403


class TestAmariParameters:
    def test_custom_n(self) -> None:
        n = AmariNeuralField(n=128)
        assert amari_state(n).shape == (128,) and n._w.shape == (128,)

    def test_tau_controls_speed(self) -> None:
        """Larger tau → slower dynamics."""
        n_fast = AmariNeuralField(tau=1.0)
        n_slow = AmariNeuralField(tau=100.0)
        I = np.ones(64) * 0.5
        n_fast.step(I)
        n_slow.step(I)
        assert np.max(np.abs(amari_state(n_fast))) > np.max(np.abs(amari_state(n_slow)))

    def test_deterministic(self) -> None:
        traces = []
        for _ in range(2):
            n = AmariNeuralField()
            I = np.ones(64) * 0.3
            trace = [n.step(I) for _ in range(100)]
            traces.append(trace)
        assert traces[0] == traces[1]

    @pytest.mark.parametrize("field", ("tau", "a_width", "b_width", "dx", "dt"))
    def test_positive_parameters_are_enforced(self, field: str) -> None:
        kwargs: Any = {field: 0.0}
        with pytest.raises(ValueError):
            AmariNeuralField(**kwargs)

    def test_failed_drive_is_atomic(self) -> None:
        neuron = AmariNeuralField(n=8)
        before = amari_state(neuron).copy()
        with pytest.raises(ValueError, match="finite"):
            neuron.step([0.0, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(neuron.u, before)
