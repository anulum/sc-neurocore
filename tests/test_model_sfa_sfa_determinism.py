# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSFADeterminism from former test_model_sfa.py

"""Focused suite: TestSFADeterminism from former test_model_sfa.py."""

from __future__ import annotations

from tests.model_sfa_support import *  # noqa: F403

class TestSFADeterminism:
    def test_bit_exact(self):
        traces = []
        for _ in range(2):
            n = SFANeuron()
            trace = [(n.step(50.0), n.v, n.g_sfa) for _ in range(300)]
            traces.append(trace)
        assert traces[0] == traces[1]
