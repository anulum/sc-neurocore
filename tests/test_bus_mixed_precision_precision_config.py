# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPrecisionConfig from former test_bus_mixed_precision.py

"""Focused suite: TestPrecisionConfig from former test_bus_mixed_precision.py."""

from __future__ import annotations

from tests.bus_mixed_precision_support import *  # noqa: F403


class TestPrecisionConfig:
    """Test the PrecisionConfig dataclass."""

    def test_q88_properties(self) -> None:
        """Q8.8 should expose sign-inclusive label and magnitude integer bits."""
        cfg = PrecisionConfig(16, 8)
        assert cfg.int_bits == 7
        assert cfg.q_label == "Q8.8"
        assert cfg.resolution == pytest.approx(1 / 256)

    def test_unsigned(self) -> None:
        """Unsigned config should have min=0 and doubled positive range."""
        cfg = PrecisionConfig(16, 8, signed=False)
        assert cfg.min_value == 0.0
        assert cfg.max_value > 200

    def test_can_represent(self) -> None:
        """Range checking should work."""
        cfg = PrecisionConfig(16, 8)
        assert cfg.can_represent(50.0)
        assert not cfg.can_represent(200.0)

    def test_encode(self) -> None:
        """Encoding should produce correct Q-format integers."""
        cfg = PrecisionConfig(16, 8)
        assert cfg.encode(1.0) == 256
        assert cfg.encode(-1.0) == -256
        assert cfg.encode(0.5) == 128
