# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Sensitivity analysis

"""Measure per-layer sensitivity to bitstream length reduction."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

_FloatArray = npt.NDArray[np.float64]


def analyze_sensitivity(
    layer_weights: list[np.ndarray[Any, Any]],
    lengths: list[int] | None = None,
    n_trials: int = 100,
    seed: int = 42,
) -> list[float]:
    """Measure per-layer sensitivity to bitstream length reduction.

    Parameters
    ----------
    layer_weights:
        One-dimensional or two-dimensional layer weight arrays. Vector weights
        are treated as a single-output dense layer.
    lengths:
        Candidate stochastic bitstream lengths to sample. When omitted, the
        estimator uses the default production planning ladder.
    n_trials:
        Number of independent input-vector samples per layer.
    seed:
        Deterministic NumPy random seed used for reproducible planning.

    Returns
    -------
    list[float]
        One non-negative sensitivity score for each supplied layer.

    Raises
    ------
    ValueError
        If trial count, candidate lengths, or layer weight arrays are invalid.
    """
    candidate_lengths = _validate_lengths(lengths)
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")

    rng = np.random.RandomState(seed)
    sensitivities: list[float] = []

    for raw_weights in layer_weights:
        weights = _as_weight_matrix(raw_weights)
        n_outputs, n_inputs = weights.shape
        clipped_weights = np.clip(weights, 0.0, 1.0)
        errors: list[float] = []

        for _ in range(n_trials):
            input_probabilities = rng.random_sample(n_inputs).astype(np.float64)
            exact = weights @ input_probabilities
            target = np.clip(exact, 0.0, None)

            length_errors: list[float] = []
            for bitstream_length in candidate_lengths:
                sc_results: list[_FloatArray] = []
                for _ in range(5):
                    bits_x = (
                        rng.random_sample((bitstream_length, n_inputs)) < input_probabilities
                    ).astype(np.float64)
                    bits_w = (
                        rng.random_sample((bitstream_length, n_outputs, n_inputs))
                        < clipped_weights[np.newaxis, :, :]
                    ).astype(np.float64)
                    and_result = bits_x[:, np.newaxis, :] * bits_w
                    counts = np.sum(
                        and_result,
                        axis=(0, 2),
                        dtype=np.float64,
                        initial=0.0,
                    )
                    sc_results.append(
                        np.asarray(counts / float(bitstream_length), dtype=np.float64)
                    )

                sc_mean = np.mean(np.stack(sc_results, axis=0), axis=0)
                err = np.mean(np.abs(sc_mean - target))
                length_errors.append(float(err))

            sensitivity = max(length_errors) - min(length_errors) if length_errors else 0.0
            errors.append(sensitivity)

        sensitivities.append(float(np.mean(errors)))

    return sensitivities


def _validate_lengths(lengths: list[int] | None) -> list[int]:
    """Return positive integer candidate bitstream lengths."""
    candidate_lengths = [32, 64, 128, 256, 512, 1024] if lengths is None else lengths
    if not candidate_lengths:
        raise ValueError("lengths must not be empty")
    for bitstream_length in candidate_lengths:
        if isinstance(bitstream_length, bool) or bitstream_length <= 0:
            raise ValueError("lengths must contain positive integers")
    return candidate_lengths


def _as_weight_matrix(weights: np.ndarray[Any, Any]) -> _FloatArray:
    """Return finite vector or matrix weights as a two-dimensional matrix."""
    matrix = np.asarray(weights, dtype=np.float64)
    if matrix.ndim not in {1, 2}:
        raise ValueError("layer weights must be one-dimensional or two-dimensional")
    if matrix.size == 0 or 0 in matrix.shape:
        raise ValueError("layer weights must not be empty")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("layer weights must be finite")
    if matrix.ndim == 1:
        return matrix.reshape(1, -1)
    return matrix
