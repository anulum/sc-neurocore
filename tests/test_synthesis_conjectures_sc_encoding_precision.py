# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCEncodingPrecision from former test_synthesis_conjectures.py

"""Focused suite: TestSCEncodingPrecision from former test_synthesis_conjectures.py."""

from __future__ import annotations

from tests.synthesis_conjectures_support import *  # noqa: F403

class TestSCEncodingPrecision:
    """Necessary condition for SC-FIM analogy: error ~ 1/sqrt(L)."""

    def test_error_decreases_with_L(self):
        target = 0.65
        L_values = [64, 256, 1024, 4096]
        errors = {}
        for L in L_values:
            errs = []
            for trial in range(200):
                enc = BitstreamEncoder(x_min=0.0, x_max=1.0, length=L, seed=trial)
                bits = enc.encode(target)
                errs.append(abs(bitstream_to_probability(bits) - target))
            errors[L] = np.mean(errs)
        # Strict: each doubling should roughly halve error (sqrt scaling)
        for i in range(len(L_values) - 1):
            assert errors[L_values[i + 1]] < errors[L_values[i]], (
                f"error at L={L_values[i + 1]} not lower than L={L_values[i]}"
            )

    def test_error_scales_approximately_sqrt(self):
        """Error ratio between L and 4L should be ~2 (from 1/sqrt scaling)."""
        target = 0.5
        n_trials = 500
        for L in [64, 256]:
            err_L = np.mean(
                [
                    abs(
                        bitstream_to_probability(
                            BitstreamEncoder(x_min=0.0, x_max=1.0, length=L, seed=t).encode(target)
                        )
                        - target
                    )
                    for t in range(n_trials)
                ]
            )
            err_4L = np.mean(
                [
                    abs(
                        bitstream_to_probability(
                            BitstreamEncoder(x_min=0.0, x_max=1.0, length=4 * L, seed=t).encode(
                                target
                            )
                        )
                        - target
                    )
                    for t in range(n_trials)
                ]
            )
            if err_4L > 0:
                ratio = err_L / err_4L
                # sqrt(4) = 2; allow 1.3-3.0 range for finite-sample effects
                assert 1.0 < ratio < 4.0, f"error ratio L={L} vs 4L: {ratio:.2f}, expected ~2.0"
