# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSiegertTransfer from former test_model_siegert.py

"""Focused suite: TestSiegertTransfer from former test_model_siegert.py."""

from __future__ import annotations

from tests.model_siegert_support import *  # noqa: F403


class TestSiegertTransfer:
    def test_returns_rate(self) -> None:
        from sc_neurocore.neurons.models.siegert import SiegertTransferFunction

        n = SiegertTransferFunction()
        rate = n.step(5.0)
        assert isinstance(rate, float)
        assert rate >= 0.0

    def test_higher_input_higher_rate(self) -> None:
        from sc_neurocore.neurons.models.siegert import SiegertTransferFunction

        n = SiegertTransferFunction()
        r_low = n.step(1.0)
        r_high = n.step(30.0)
        assert r_high >= r_low
