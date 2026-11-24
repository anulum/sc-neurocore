# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSeedContracts from former test_fault_injection_module.py

"""Focused suite: TestSeedContracts from former test_fault_injection_module.py."""

from __future__ import annotations

from typing import Any, cast

import pytest
from sc_neurocore.fault_injection.fault_injection import (
    FaultInjector,
    FaultModel,
    ResilienceBenchmark,
)


class TestSeedContracts:
    """Checks on the injector seed."""

    def test_fault_injector_reproducible_with_same_seed(self) -> None:
        """The same seed injects the same faults."""
        bits = [0, 1, 1, 0, 1, 0, 1, 1]
        import numpy as np

        bitstream = np.array(bits, dtype=np.uint8)
        a, a_flipped = FaultInjector(seed=7).inject(bitstream, model=FaultModel.BIT_FLIP, ber=0.2)
        b, b_flipped = FaultInjector(seed=7).inject(bitstream, model=FaultModel.BIT_FLIP, ber=0.2)
        assert a_flipped == b_flipped
        assert np.array_equal(a, b)

    @pytest.mark.parametrize("seed", [1.5, "7", True])
    def test_rejects_non_integer_seed(self, seed: object) -> None:
        """A seed that is not an integer is rejected."""
        with pytest.raises(ValueError, match="seed"):
            FaultInjector(seed=cast(Any, seed))
        with pytest.raises(ValueError, match="seed"):
            ResilienceBenchmark(seed=cast(Any, seed))
