# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFixedPointComparison from former test_fault_injection.py

"""Focused suite: TestFixedPointComparison from former test_fault_injection.py."""

from __future__ import annotations

from tests.fault_injection_support import *  # noqa: F403


class TestFixedPointComparison:
    def test_sc_beats_fp_at_10pct_error(self):
        """SC should degrade less than 16-bit fixed-point at 10% BER."""
        target = 0.65
        rate = 0.1
        L = 2000
        n_trials = 200
        rng = np.random.default_rng(42)

        sc_errs, fp_errs = [], []
        for trial in range(n_trials):
            # SC
            bits = generate_bernoulli_bitstream(target, L, rng=RNG(trial))
            faulted = FaultInjector.inject_bit_flips(bits, rate)
            sc_errs.append(abs(np.mean(faulted) - target))

            # Fixed-point 16-bit
            fp_val = int(target * (1 << 16))
            fp_bits = np.array([(fp_val >> b) & 1 for b in range(16)])
            flip = rng.random(16) < rate
            fp_faulted = fp_bits ^ flip.astype(int)
            decoded = sum(b << i for i, b in enumerate(fp_faulted)) / (1 << 16)
            fp_errs.append(abs(decoded - target))

        assert np.mean(sc_errs) < np.mean(fp_errs), (
            f"SC mean err {np.mean(sc_errs):.4f} >= FP {np.mean(fp_errs):.4f}"
        )
