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


def analyze_sensitivity(
    layer_weights: list[np.ndarray[Any, Any]],
    lengths: list[int] | None = None,
    n_trials: int = 100,
    seed: int = 42,
) -> list[float]:
    """Measure per-layer sensitivity to bitstream length reduction."""
    if lengths is None:
        lengths = [32, 64, 128, 256, 512, 1024]

    rng = np.random.RandomState(seed)
    sensitivities = []

    for w in layer_weights:
        n_in = w.shape[1] if w.ndim == 2 else w.shape[0]
        errors = []

        for _ in range(n_trials):
            x = rng.random(n_in)
            exact = x @ w.T if w.ndim == 2 else x * w

            length_errors = []
            for L in lengths:
                sc_results = []
                for trial in range(5):
                    bits_x = (rng.random((L, n_in)) < x).astype(np.float64)
                    if w.ndim == 2:
                        n_out = w.shape[0]
                        bits_w = np.zeros((L, n_out, n_in))
                        for j in range(n_out):
                            w_prob = np.clip(w[j], 0, 1)
                            bits_w[:, j, :] = (rng.random((L, n_in)) < w_prob).astype(np.float64)
                        and_result = bits_x[:, np.newaxis, :] * bits_w
                        sc_out = and_result.sum(axis=(0, 2)) / L
                    else:  # pragma: no cover
                        w_prob = np.clip(w, 0, 1)
                        bits_w = (rng.random((L,)) < w_prob).astype(np.float64)
                        sc_out = (bits_x.mean(axis=0) * bits_w).mean()
                    sc_results.append(sc_out)

                sc_mean = np.mean(sc_results, axis=0)
                err = np.mean(np.abs(sc_mean - np.clip(exact, 0, None)))
                length_errors.append(err)

            sensitivity = max(length_errors) - min(length_errors) if length_errors else 0.0
            errors.append(sensitivity)

        sensitivities.append(float(np.mean(errors)))

    return sensitivities
