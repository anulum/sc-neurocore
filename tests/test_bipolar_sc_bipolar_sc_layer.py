# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBipolarSCLayer from former test_bipolar_sc.py

"""Focused suite: TestBipolarSCLayer from former test_bipolar_sc.py."""

from __future__ import annotations

from tests.bipolar_sc_support import *  # noqa: F403

class TestBipolarSCLayer:
    def test_output_shape(self):
        inputs = np.array([0.5, -0.3, 0.1])
        weights = np.array([[0.2, 0.4, -0.1], [-0.5, 0.3, 0.8]])
        out = bipolar_sc_layer(inputs, weights, bias=None, L=1000)
        assert out.shape == (2,)

    def test_relu_clips_negative(self):
        inputs = np.array([-0.9])
        weights = np.array([[0.9]])
        out = bipolar_sc_layer(inputs, weights, bias=None, L=50000, activation="relu")
        # -0.9 * 0.9 = -0.81, relu -> 0
        assert out[0] >= 0.0

    def test_output_bounded(self):
        inputs = np.random.default_rng(42).uniform(-1, 1, 10)
        weights = np.random.default_rng(43).uniform(-1, 1, (5, 10))
        out = bipolar_sc_layer(inputs, weights, bias=None, L=1000)
        assert (out >= -1.0).all() and (out <= 1.0).all()

    def test_rejects_unknown_activation(self):
        with pytest.raises(ValueError, match="activation"):
            bipolar_sc_layer(
                np.array([0.5]),
                np.array([[0.5]]),
                bias=None,
                L=1000,
                activation="sigmoid",
            )
