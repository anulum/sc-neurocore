# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAstrocytePerformance from former test_model_astrocyte.py

"""Focused suite: TestAstrocytePerformance from former test_model_astrocyte.py."""

from __future__ import annotations

from tests.model_astrocyte_support import *  # noqa: F403


class TestAstrocytePerformance:
    def test_isolation_throughput(self):
        n = AstrocyteModel()
        N = 20000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(0.5)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 20000
