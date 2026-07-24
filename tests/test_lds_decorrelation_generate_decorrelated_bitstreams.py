# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGenerateDecorrelatedBitstreams from former test_lds_decorrelation.py

"""Focused suite: TestGenerateDecorrelatedBitstreams from former test_lds_decorrelation.py."""

from __future__ import annotations

from tests.lds_decorrelation_support import *  # noqa: F403


class TestGenerateDecorrelatedBitstreams:
    def test_output_shape_1d(self):
        probs = np.array([0.3, 0.5, 0.7])
        streams = generate_decorrelated_bitstreams(probs, length=128)
        assert streams.shape == (3, 128)

    def test_output_shape_2d(self):
        probs = np.array([[0.3, 0.5], [0.7, 0.2]])
        streams = generate_decorrelated_bitstreams(probs, length=256)
        assert streams.shape == (2, 2, 256)

    def test_output_binary(self):
        probs = np.array([0.4, 0.6])
        streams = generate_decorrelated_bitstreams(probs, length=100)
        assert set(np.unique(streams)).issubset({0, 1})

    def test_probability_convergence(self):
        """Bitstream means should converge to target probabilities."""
        probs = np.array([0.2, 0.5, 0.8])
        streams = generate_decorrelated_bitstreams(probs, length=4096, method="sobol")
        means = streams.mean(axis=-1)
        np.testing.assert_allclose(means, probs, atol=0.05)

    def test_sobol_method(self):
        probs = np.array([0.5])
        streams = generate_decorrelated_bitstreams(probs, length=64, method="sobol")
        assert streams.shape == (1, 64)

    def test_halton_method(self):
        probs = np.array([0.5])
        streams = generate_decorrelated_bitstreams(probs, length=64, method="halton")
        assert streams.shape == (1, 64)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            generate_decorrelated_bitstreams(np.array([0.5]), method="random")

    def test_decorrelation(self):
        """Different dimensions of the LDS should be decorrelated."""
        probs = np.array([0.5, 0.5])
        streams = generate_decorrelated_bitstreams(probs, length=2048, method="sobol")
        # Cross-correlation between the two streams should be near zero
        corr = np.corrcoef(streams[0].astype(float), streams[1].astype(float))[0, 1]
        assert abs(corr) < 0.1

    def test_empty_probs(self):
        probs = np.array([])
        streams = generate_decorrelated_bitstreams(probs, length=100)
        assert streams.shape == (0, 100)

    def test_zero_and_one(self):
        probs = np.array([0.0, 1.0])
        streams = generate_decorrelated_bitstreams(probs, length=100, method="sobol")
        assert np.all(streams[0] == 0)
        assert np.all(streams[1] == 1)
