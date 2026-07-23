# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMixedPrecisionSpec from former test_bus_mixed_precision.py

"""Focused suite: TestMixedPrecisionSpec from former test_bus_mixed_precision.py."""

from __future__ import annotations

from tests.bus_mixed_precision_support import *  # noqa: F403

class TestMixedPrecisionSpec:
    """Test the mixed-precision specification."""

    def test_total_bits(self) -> None:
        """Total bits should sum correctly."""
        spec = MixedPrecisionSpec(
            {
                "v": PrecisionConfig(16, 8),
                "u": PrecisionConfig(8, 4),
            }
        )
        assert spec.total_bits == 24

    def test_variables(self) -> None:
        """Should list all variables."""
        spec = MixedPrecisionSpec(
            {
                "v": PrecisionConfig(16, 8),
                "u": PrecisionConfig(8, 4),
            }
        )
        assert set(spec.variables) == {"v", "u"}

    def test_get(self) -> None:
        """Should retrieve config by name."""
        spec = MixedPrecisionSpec(
            {
                "v": PrecisionConfig(16, 8),
            }
        )
        assert spec.get("v").data_width == 16
        assert spec.get("v").q_label == "Q8.8"

    def test_get_missing(self) -> None:
        """Should raise on missing variable."""
        spec = MixedPrecisionSpec({"v": PrecisionConfig(16, 8)})
        with pytest.raises(KeyError, match="not in"):
            spec.get("w")

    def test_summary(self) -> None:
        """Summary should be human-readable."""
        spec = MixedPrecisionSpec(
            {
                "v": PrecisionConfig(16, 8),
                "u": PrecisionConfig(8, 4),
            }
        )
        s = spec.summary()
        assert "24 bits total" in s
        assert "Q8.8" in s
        assert "Q4.4" in s
