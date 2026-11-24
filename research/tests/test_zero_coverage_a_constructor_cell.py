# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestConstructorCell from former test_zero_coverage_a.py

"""Focused suite: TestConstructorCell from former test_zero_coverage_a.py."""

from __future__ import annotations

import numpy as np


class TestConstructorCell:
    """The self-replicating constructor cell research module."""

    def test_replicate_and_mutate(self) -> None:
        """Replication returns a constructor cell (the check is on the type only)."""
        from exotic.constructor import ConstructorCell

        bp = np.random.randn(8)
        c = ConstructorCell(id=0, blueprint=bp)
        clone = c.replicate()
        assert isinstance(clone, ConstructorCell)
        c.mutate_blueprint(rate=0.1)
