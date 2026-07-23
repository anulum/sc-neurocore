# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSiegertIsolation from former test_model_siegert.py

"""Focused suite: TestSiegertIsolation from former test_model_siegert.py."""

from __future__ import annotations

from tests.model_siegert_support import *  # noqa: F403

class TestSiegertIsolation:
    def test_defaults(self) -> None:
        n = SiegertTransferFunction()
        assert n.tau_m == 20.0 and n.tau_rp == 2.0
        assert n.v_threshold == -50.0 and n.v_reset == -70.0

    def test_step_returns_float(self) -> None:
        n = SiegertTransferFunction()
        assert isinstance(n.step(20.0), (float, np.floating))

    def test_reset_noop(self) -> None:
        n = SiegertTransferFunction()
        n.step(20.0)
        n.reset()  # should not raise
