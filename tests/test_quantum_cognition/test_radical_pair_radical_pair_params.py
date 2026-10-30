# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRadicalPairParams from former test_radical_pair.py

"""Focused suite: TestRadicalPairParams from former test_radical_pair.py."""

from __future__ import annotations

from tests.test_quantum_cognition.radical_pair_support import *  # noqa: F403


class TestRadicalPairParams:
    """Default and custom radical-pair parameters."""

    def test_defaults(self) -> None:
        """Defaults are 10 MHz hyperfine, 1 MHz exchange and a 100 us lifetime."""
        p = RadicalPairParams()
        assert p.hyperfine_a == 10.0
        assert p.exchange_j == 1.0
        assert p.lifetime_us == 100.0

    def test_custom(self) -> None:
        """Supplied hyperfine and exchange values are kept."""
        p = RadicalPairParams(hyperfine_a=50.0, exchange_j=5.0)
        assert p.hyperfine_a == 50.0
