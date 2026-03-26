# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for photonic stochastic computing layer

import numpy as np
import pytest

from sc_neurocore.optics.photonic_layer import PhotonicBitstreamLayer


class TestPhotonicBitstreamLayer:
    def test_interference_shape(self):
        layer = PhotonicBitstreamLayer(n_channels=4)
        intensity = layer.simulate_interference(length=100)
        assert intensity.shape == (4, 100)

    def test_interference_range(self):
        layer = PhotonicBitstreamLayer(n_channels=8)
        intensity = layer.simulate_interference(length=10000)
        assert intensity.min() >= 0.0
        assert intensity.max() <= 1.0

    def test_forward_shape(self):
        layer = PhotonicBitstreamLayer(n_channels=3)
        probs = np.array([0.3, 0.5, 0.7])
        bits = layer.forward(probs, length=1024)
        assert bits.shape == (3, 1024)
        assert bits.dtype == np.uint8
        assert set(np.unique(bits)).issubset({0, 1})

    def test_forward_probability_scaling(self):
        np.random.seed(42)
        layer = PhotonicBitstreamLayer(n_channels=2)
        probs = np.array([0.2, 0.8])
        bits = layer.forward(probs, length=50000)
        # Channel with p=0.2 should fire less than p=0.8
        assert bits[0].mean() < bits[1].mean()

    def test_wrong_input_shape(self):
        layer = PhotonicBitstreamLayer(n_channels=3)
        with pytest.raises(ValueError, match="does not match"):
            layer.forward(np.array([0.5, 0.5]), length=100)

    def test_zero_probability(self):
        np.random.seed(0)
        layer = PhotonicBitstreamLayer(n_channels=1)
        bits = layer.forward(np.array([0.0]), length=1000)
        assert bits.mean() == 0.0

    def test_one_probability(self):
        np.random.seed(0)
        layer = PhotonicBitstreamLayer(n_channels=1)
        bits = layer.forward(np.array([1.0]), length=1000)
        assert bits.mean() == 1.0
