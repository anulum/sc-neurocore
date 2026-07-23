# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSobolConvergence from former test_sc_convergence.py

"""Focused suite: TestSobolConvergence from former test_sc_convergence.py."""

from __future__ import annotations

from tests.sc_convergence_support import *  # noqa: F403

class TestSobolConvergence:
    """Sobol quasi-random should converge faster than Bernoulli."""

    def test_sobol_lower_error_at_same_length(self):
        target = 0.65
        L = 512
        bern_errs, sobol_errs = [], []
        for trial in range(N_TRIALS):
            b = generate_bernoulli_bitstream(target, L, rng=RNG(trial))
            bern_errs.append(abs(np.mean(b) - target))
            s = generate_sobol_bitstream(target, L, seed=trial)
            sobol_errs.append(abs(np.mean(s) - target))
        assert np.mean(sobol_errs) < np.mean(bern_errs), "Sobol should beat Bernoulli"

    def test_sobol_output_binary(self):
        bits = generate_sobol_bitstream(0.5, 1024, seed=42)
        assert set(np.unique(bits)).issubset({0, 1})

    def test_sobol_probability_accurate(self):
        for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
            bits = generate_sobol_bitstream(p, 4096, seed=42)
            np.testing.assert_allclose(np.mean(bits), p, atol=0.02)
