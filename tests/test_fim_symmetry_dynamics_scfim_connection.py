# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCFIMConnection from former test_fim_symmetry_dynamics.py

"""Focused suite: TestSCFIMConnection from former test_fim_symmetry_dynamics.py."""

from __future__ import annotations

from tests.fim_symmetry_dynamics_support import *  # noqa: F403

class TestSCFIMConnection:
    def test_longer_bitstream_higher_precision(self):
        """Longer bitstream L should give lower SC encoding error.
        This is the necessary condition for the SC-FIM conjecture."""
        from sc_neurocore import BitstreamEncoder, bitstream_to_probability

        errors = {}
        target_p = 0.65
        for L in [64, 256, 1024]:
            trial_errors = []
            for trial in range(100):
                enc = BitstreamEncoder(x_min=0.0, x_max=1.0, length=L, seed=trial)
                bits = enc.encode(target_p)
                recovered = bitstream_to_probability(bits)
                trial_errors.append(abs(recovered - target_p))
            errors[L] = np.mean(trial_errors)

        # Longer L → lower error (necessary for SC-FIM)
        assert errors[1024] < errors[64], (
            f"L=1024 error {errors[1024]:.4f} >= L=64 error {errors[64]:.4f}"
        )
