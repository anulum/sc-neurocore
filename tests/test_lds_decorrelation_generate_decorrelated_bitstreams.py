# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Matrix LDS thresholding and finite covariance contracts

"""Check LDS bitstream shapes, thresholded means and finite covariance."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.utils.lds_decorrelation import generate_decorrelated_bitstreams


class TestGenerateDecorrelatedBitstreams:
    """Exercise the public matrix encoder across its existing input shapes."""

    def test_output_shape_1d(self) -> None:
        """Preserve one encoded row per probability."""
        probs = np.array([0.3, 0.5, 0.7])
        streams = generate_decorrelated_bitstreams(probs, length=128)
        assert streams.shape == (3, 128)

    def test_output_shape_2d(self) -> None:
        """Restore matrix axes before the bitstream axis."""
        probs = np.array([[0.3, 0.5], [0.7, 0.2]])
        streams = generate_decorrelated_bitstreams(probs, length=256)
        assert streams.shape == (2, 2, 256)

    def test_output_binary(self) -> None:
        """Return only binary samples for finite probabilities."""
        probs = np.array([0.4, 0.6])
        streams = generate_decorrelated_bitstreams(probs, length=100)
        assert set(np.unique(streams)).issubset({0, 1})

    def test_probability_convergence(self) -> None:
        """Bitstream means should converge to target probabilities."""
        probs = np.array([0.2, 0.5, 0.8])
        streams = generate_decorrelated_bitstreams(probs, length=4096, method="sobol")
        means = streams.mean(axis=-1)
        np.testing.assert_allclose(means, probs, atol=0.05)

    def test_sobol_method(self) -> None:
        """Exercise the selected Sobol sampler through public encoding."""
        probs = np.array([0.5])
        streams = generate_decorrelated_bitstreams(probs, length=64, method="sobol")
        assert streams.shape == (1, 64)

    def test_halton_method(self) -> None:
        """Exercise the selected Halton sampler through public encoding."""
        probs = np.array([0.5])
        streams = generate_decorrelated_bitstreams(probs, length=64, method="halton")
        assert streams.shape == (1, 64)

    def test_unknown_method_raises(self) -> None:
        """Reject an unknown sequence method before encoding output."""
        with pytest.raises(ValueError, match="Unknown method"):
            generate_decorrelated_bitstreams(np.array([0.5]), method="random")

    def test_decorrelation(self) -> None:
        """Different dimensions of the LDS should be decorrelated."""
        probs = np.array([0.5, 0.5])
        streams = generate_decorrelated_bitstreams(probs, length=2048, method="sobol")
        # Cross-correlation between the two streams should be near zero
        corr = np.corrcoef(streams[0].astype(float), streams[1].astype(float))[0, 1]
        assert abs(corr) < 0.1

    def test_empty_probs(self) -> None:
        """Retain the requested bit length for an empty probability axis."""
        probs = np.array([])
        streams = generate_decorrelated_bitstreams(probs, length=100)
        assert streams.shape == (0, 100)

    def test_zero_and_one(self) -> None:
        """Encode the probability endpoints as exact constant bitstreams."""
        probs = np.array([0.0, 1.0])
        streams = generate_decorrelated_bitstreams(probs, length=100, method="sobol")
        assert np.all(streams[0] == 0)
        assert np.all(streams[1] == 1)


@pytest.mark.parametrize(
    "method, expected_covariance", [("sobol", 1.0 / 16), ("halton", -1.0 / 32)]
)
def test_finite_lds_dimensions_retain_observed_nonzero_covariance(
    method: str, expected_covariance: float
) -> None:
    """Retain the complete seeded streams behind the finite-covariance counterexample.

    Parameters
    ----------
    method
        Selected public Sobol or Halton sequence family.
    expected_covariance
        Population covariance of the two retained eight-sample binary streams.
    """
    probabilities = np.array([[0.3, 0.7]], dtype=np.float64)
    probabilities.setflags(write=False)
    before = probabilities.tobytes()
    streams = generate_decorrelated_bitstreams(probabilities, length=8, method=method, seed=42)
    assert streams.shape == (1, 2, 8) and streams.dtype == np.uint8
    if method == "sobol":
        expected = [[[0, 0, 0, 1, 1, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0, 1]]]
    else:
        expected = [[[0, 1, 0, 0, 0, 1, 0, 0], [1, 0, 1, 1, 0, 1, 1, 0]]]
    np.testing.assert_array_equal(streams, expected)
    a, b = streams[0]
    count_a, count_b = int(a.sum()), int(b.sum())
    joint = int(np.logical_and(a, b).sum())
    covariance = (joint * 8 - count_a * count_b) / 64
    assert covariance == expected_covariance and covariance != 0.0
    assert probabilities.tobytes() == before and not probabilities.flags.writeable
