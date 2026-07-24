# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPernarowskiDeterminism from former test_model_pernarowski.py

"""Focused suite: TestPernarowskiDeterminism from former test_model_pernarowski.py."""

from __future__ import annotations

from tests.model_pernarowski_support import *  # noqa: F403


class TestPernarowskiDeterminism:
    def test_bit_exact_reproducibility(self):
        """Identical runs produce identical traces (no RNG)."""
        traces = []
        for _ in range(2):
            n = PernarowskiNeuron()
            trace = [(n.step(0.5), n.v, n.w, n.z) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]
