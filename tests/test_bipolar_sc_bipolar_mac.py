# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBipolarMAC from former test_bipolar_sc.py

"""Focused suite: TestBipolarMAC from former test_bipolar_sc.py."""

from __future__ import annotations

from tests.bipolar_sc_support import *  # noqa: F403


class TestBipolarMAC:
    def test_single_input(self):
        # 1 input: dot product = 0.5 * 0.8 = 0.4
        inputs = np.array([0.5])
        weights = np.array([[0.8]])
        result = bipolar_mac(inputs, weights, L=50000, seed=42)
        assert abs(result[0] - 0.4) < 0.05

    def test_two_inputs_dot_product(self):
        # dot product: 0.6*0.5 + (-0.4)*0.3 = 0.3 - 0.12 = 0.18
        inputs = np.array([0.6, -0.4])
        weights = np.array([[0.5, 0.3]])
        result = bipolar_mac(inputs, weights, L=50000, seed=42)
        assert abs(result[0] - 0.18) < 0.1

    def test_multiple_outputs(self):
        inputs = np.array([0.5, -0.5])
        weights = np.array([[0.8, 0.2], [-0.3, 0.7]])
        result = bipolar_mac(inputs, weights, L=50000, seed=42)
        assert result.shape == (2,)
        # out[0] = 0.5*0.8 + (-0.5)*0.2 = 0.4 - 0.1 = 0.3
        # out[1] = 0.5*(-0.3) + (-0.5)*0.7 = -0.15 - 0.35 = -0.5
        assert abs(result[0] - 0.3) < 0.1
        assert abs(result[1] - (-0.5)) < 0.1

    def test_longer_bitstream_more_accurate(self):
        inputs = np.array([0.5])
        weights = np.array([[0.8]])
        r1 = bipolar_mac(inputs, weights, L=1000, seed=42)
        r2 = bipolar_mac(inputs, weights, L=100000, seed=42)
        expected = 0.4
        assert abs(r2[0] - expected) < abs(r1[0] - expected) + 0.01

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape"):
            bipolar_mac(np.array([0.5, -0.2]), np.array([[0.8]]), L=1000, seed=42)

    def test_rejects_out_of_range_inputs_and_weights(self):
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            bipolar_mac(np.array([1.2]), np.array([[0.8]]), L=1000, seed=42)
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            bipolar_mac(np.array([0.5]), np.array([[1.2]]), L=1000, seed=42)
