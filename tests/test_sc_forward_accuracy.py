# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAccuracy from former test_sc_forward.py

"""Focused suite: TestAccuracy from former test_sc_forward.py."""

from __future__ import annotations

from tests.sc_forward_support import *  # noqa: F403


class TestAccuracy:
    """NEU-SCPN.4 — sc_forward estimates W @ probs within stochastic tolerance."""

    def test_single_product_within_three_sigma(self) -> None:
        length = 4096
        weights = np.array([[0.4]])
        probs = np.array([0.7])
        packed = _pack_weights(weights, length, seed=0x1357)
        estimate = sc_forward_numpy(packed, probs, length, seed=0xACE1)
        reference = float(weights[0, 0] * probs[0])
        tolerance = 3.0 * np.sqrt(reference * (1.0 - reference) / length)
        # LFSR discretisation adds a small deterministic bias beyond the 3-sigma band.
        assert abs(estimate[0] - reference) <= tolerance + 0.005

    def test_network_within_tolerance(self) -> None:
        rng = np.random.default_rng(20260621)
        n_out, n_in, length = 8, 32, 4096
        weights = rng.random((n_out, n_in))
        probs = rng.random(n_in)
        packed = _pack_weights(weights, length, seed=0x2468)
        estimate = sc_forward(packed, probs, length=length, seed=0xACE1)
        reference = weights @ probs
        per_product = weights * probs
        variance = np.maximum(per_product * (1.0 - per_product), 0.0).sum(axis=1)
        tolerance = 3.0 * np.sqrt(variance / length) + 0.02
        npt.assert_array_less(np.abs(estimate - reference), tolerance)

    def test_seed_zero_uses_non_zero_lfsr_seed(self) -> None:
        # base seed 0 forces the per-input seed-zero guard.
        packed = _pack_weights(np.array([[0.5, 0.5]]), 1024, seed=0x99)
        estimate = sc_forward_numpy(packed, np.array([0.5, 0.5]), 1024, seed=0)
        assert np.isfinite(estimate).all()
