# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRadicalPairParams from former test_radical_pair.py

"""Focused suite: TestRadicalPairParams from former test_radical_pair.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent))
from radical_pair_support import *  # noqa: F403

class TestRadicalPairParams:
    def test_defaults(self) -> None:
        p = RadicalPairParams()
        assert p.hyperfine_a == 10.0
        assert p.exchange_j == 1.0
        assert p.lifetime_us == 100.0

    def test_custom(self) -> None:
        p = RadicalPairParams(hyperfine_a=50.0, exchange_j=5.0)
        assert p.hyperfine_a == 50.0
