# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFromPreset from former test_bus_mixed_precision.py

"""Focused suite: TestFromPreset from former test_bus_mixed_precision.py."""

from __future__ import annotations

from tests.bus_mixed_precision_support import *  # noqa: F403


class TestFromPreset:
    """Test preset-based mixed-precision creation."""

    def test_basic(self) -> None:
        """Should create spec from named presets."""
        spec = from_preset({"v": "q88", "u": "q44"})
        assert spec.get("v").data_width == 16
        assert spec.get("u").data_width == 8

    def test_all_presets_valid(self) -> None:
        """Every preset in the registry should be valid."""
        for name, cfg in PRECISION_PRESETS.items():
            assert cfg.data_width > 0
            assert cfg.fraction >= 0
            assert cfg.fraction < cfg.data_width

    def test_unknown_preset(self) -> None:
        """Should raise on unknown preset."""
        with pytest.raises(KeyError, match="Unknown preset"):
            from_preset({"v": "q999"})

    def test_case_insensitive(self) -> None:
        """Preset lookup should be case-insensitive."""
        spec = from_preset({"v": "Q7.8"})
        assert spec.get("v").data_width == 15
        assert spec.get("v").q_label == "Q7.8"

    def test_block_floating_preset(self) -> None:
        """Block-floating presets should be materializable and typed."""
        spec = from_preset({"k": "bfp16e3x32"})
        cfg = spec.get("k")
        assert isinstance(cfg, BlockFloatingPrecisionConfig)
        assert cfg.mantissa_bits == 16
        assert cfg.exponent_bits == 3
        assert cfg.block_size == 32
        assert cfg.kind == "block_floating"
