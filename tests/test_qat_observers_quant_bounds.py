# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQuantBounds from former test_qat_observers.py

"""Focused suite: TestQuantBounds from former test_qat_observers.py."""

from __future__ import annotations

import pytest
from tests.qat_observers_support import (
    _quant_bounds,
)


class TestQuantBounds:
    """Signed and unsigned integer bounds for a bit width."""

    def test_signed_bounds(self) -> None:
        """Signed 8-bit and 4-bit bounds are (-128, 127) and (-8, 7)."""
        assert _quant_bounds(8, unsigned=False) == (-128, 127)
        assert _quant_bounds(4, unsigned=False) == (-8, 7)

    def test_unsigned_bounds(self) -> None:
        """Unsigned 8-bit and 4-bit bounds are (0, 255) and (0, 15)."""
        assert _quant_bounds(8, unsigned=True) == (0, 255)
        assert _quant_bounds(4, unsigned=True) == (0, 15)

    def test_rejects_low_bits(self) -> None:
        """A bit width below the supported minimum is refused."""
        with pytest.raises(ValueError, match="n_bits must be >= 2"):
            _quant_bounds(1, unsigned=False)
