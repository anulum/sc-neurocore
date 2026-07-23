# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPinskyRinzelNumerics from former test_model_pinsky_rinzel.py

"""Focused suite: TestPinskyRinzelNumerics from former test_model_pinsky_rinzel.py."""

from __future__ import annotations

from tests.model_pinsky_rinzel_support import *  # noqa: F403

class TestPinskyRinzelNumerics:
    def test_bit_exact_reproducibility(self):
        def trace() -> list[tuple[int, float]]:
            n = PinskyRinzelNeuron()
            return [(n.step(30.0), n.v_s) for _ in range(500)]

        assert trace() == trace()

    @pytest.mark.parametrize("dt", [0.01, 0.02, 0.05])
    def test_dt_stability(self, dt: float):
        n = PinskyRinzelNeuron(dt=dt)
        for _ in range(20000):
            n.step(30.0)
        assert np.isfinite(n.v_s) and np.isfinite(n.v_d)
