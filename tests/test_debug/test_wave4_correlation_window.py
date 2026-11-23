# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCorrelationWindow from former test_wave4.py

"""Focused suite: TestCorrelationWindow from former test_wave4.py."""

from __future__ import annotations

import pytest
from sc_neurocore.debug.hil_client import (
    CorrelationWindow,
)


class TestCorrelationWindow:
    """The window over recent correlations."""

    def test_mean(self) -> None:
        """The window reports its mean correlation."""
        cw = CorrelationWindow(4)
        for v in [0.1, 0.2, 0.3, 0.4]:
            cw.add(v)
        assert cw.mean() == pytest.approx(0.25)

    def test_max(self) -> None:
        """The window reports its largest correlation."""
        cw = CorrelationWindow(4)
        for v in [0.1, 0.5, 0.2]:
            cw.add(v)
        assert cw.max() == pytest.approx(0.5)

    def test_count(self) -> None:
        """The window counts its five values."""
        cw = CorrelationWindow(10)
        for _ in range(5):
            cw.add(1.0)
        assert cw.count == 5
