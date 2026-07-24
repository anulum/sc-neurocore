# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTransferMatrixUnitarity from former test_crosstalk.py

"""Focused suite: TestTransferMatrixUnitarity from former test_crosstalk.py."""

from __future__ import annotations

from crosstalk_support import *  # noqa: F403


class TestTransferMatrixUnitarity:
    @pytest.fixture
    def model(self) -> CrosstalkModel:
        return CrosstalkModel()

    def test_transfer_matrix_is_unitary(self, model):
        pair = WaveguidePair(gap_nm=150.0, coupling_length_um=25.0)
        t = model.transfer_matrix(pair)
        # T · T† == I for any real κL
        identity = t @ t.conj().T
        assert np.allclose(identity, np.eye(2), atol=1e-12)

    def test_power_conservation_on_single_port_excitation(self, model):
        pair = WaveguidePair(gap_nm=200.0, coupling_length_um=7.0)
        p_a, p_b = model.compute_crosstalk(pair, (1.0, 0.0))
        assert p_a + p_b == pytest.approx(1.0, rel=1e-12)

    def test_power_conservation_on_two_port_excitation(self, model):
        # ``compute_crosstalk`` takes input field amplitudes (the FFI name
        # ``input_power`` is historical); output power sums must match
        # |a|² + |b|² of the input amplitude tuple under unitary evolution.
        pair = WaveguidePair(gap_nm=200.0, coupling_length_um=7.0)
        amp_a, amp_b = 0.6, 0.4
        p_a, p_b = model.compute_crosstalk(pair, (amp_a, amp_b))
        assert p_a + p_b == pytest.approx(amp_a**2 + amp_b**2, rel=1e-12)
