# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBipolarSCLayerBiasAndActivation from former test_bipolar_sc.py

"""Focused suite: TestBipolarSCLayerBiasAndActivation from former test_bipolar_sc.py."""

from __future__ import annotations

from tests.bipolar_sc_support import *  # noqa: F403

class TestBipolarSCLayerBiasAndActivation:
    """The optional-bias branch and the tanh activation of the SC layer."""

    def test_bias_is_added_to_output(self):
        inputs = np.array([0.5, -0.3])
        weights = np.array([[0.2, 0.4], [-0.5, 0.3]])
        bias = np.array([0.5, -0.5])
        out = bipolar_sc_layer(inputs, weights, bias=bias, L=2000, activation="none")
        assert out.shape == (2,)
        assert (out >= -1.0).all() and (out <= 1.0).all()

    def test_bias_shape_mismatch_rejected(self):
        with pytest.raises(ValueError, match="bias shape"):
            bipolar_sc_layer(
                np.array([0.5, -0.3]),
                np.array([[0.2, 0.4], [-0.5, 0.3]]),
                bias=np.array([0.1]),  # (1,) against a (2,) output
                L=1000,
            )

    def test_bias_non_finite_rejected(self):
        with pytest.raises(ValueError, match="NaN or Inf"):
            bipolar_sc_layer(
                np.array([0.5, -0.3]),
                np.array([[0.2, 0.4], [-0.5, 0.3]]),
                bias=np.array([np.nan, 0.0]),
                L=1000,
            )

    def test_tanh_activation_bounds_output(self):
        inputs = np.array([0.9, 0.8])
        weights = np.array([[0.9, 0.9]])
        out = bipolar_sc_layer(inputs, weights, bias=None, L=50000, activation="tanh")
        assert out.shape == (1,)
        assert -1.0 <= out[0] <= 1.0
