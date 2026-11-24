# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestInjectAtPositionsContracts from former test_fault_injection_module.py

"""Focused suite: TestInjectAtPositionsContracts from former test_fault_injection_module.py."""

from __future__ import annotations

from typing import Any, cast

import pytest
from sc_neurocore.fault_injection.fault_injection import (
    FaultInjector,
)


class TestInjectAtPositionsContracts:
    """Checks on position-targeted injection."""

    def test_flips_requested_positions(self) -> None:
        """The requested positions are flipped."""
        import numpy as np

        injector = FaultInjector(seed=1)
        bitstream = np.array([0, 1, 0, 1], dtype=np.uint8)
        out = injector.inject_at_positions(bitstream, [0, 2])
        assert out.tolist() == [1, 1, 1, 1]

    @pytest.mark.parametrize(
        ("positions", "match"),
        [
            ("0,1", "list"),
            ([0, 0], "unique"),
            ([-1], "bounds"),
            ([10], "bounds"),
            ([1.5], "integers"),
        ],
    )
    def test_rejects_invalid_positions(self, positions: object, match: str) -> None:
        """Each invalid position list is rejected."""
        import numpy as np

        injector = FaultInjector(seed=1)
        bitstream = np.array([0, 1, 0, 1], dtype=np.uint8)
        with pytest.raises(ValueError, match=match):
            injector.inject_at_positions(bitstream, cast(Any, positions))
