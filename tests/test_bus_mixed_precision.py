# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mixed-precision solver contracts

"""Tests for mixed-precision specifications, solvers, and presets."""

from __future__ import annotations

import pytest

from sc_neurocore.compiler.mixed_precision import (
    BlockFloatingPrecisionConfig,
    MixedPrecisionSpec,
    PRECISION_PRESETS,
    PrecisionConfig,
    from_preset,
    solve_precision,
)


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


class TestConstraintSolver:
    """Test the automatic precision constraint solver."""

    def test_basic_solve(self) -> None:
        """Should produce valid configs from bounds."""
        spec = solve_precision(
            bounds={"v": (-128, 127), "u": (-10, 10)},
        )
        assert spec.get("v").can_represent(-128)
        assert spec.get("v").can_represent(127)
        assert spec.get("u").can_represent(-10)
        assert spec.get("u").can_represent(10)

    def test_resolution_honoured(self) -> None:
        """Requested resolution should be achievable."""
        spec = solve_precision(
            bounds={"v": (-1, 1)},
            min_resolution={"v": 0.001},
        )
        assert spec.get("v").resolution <= 0.001

    def test_budget_constraint(self) -> None:
        """Should reduce precision to fit bit budget."""
        spec = solve_precision(
            bounds={"v": (-128, 127), "u": (-10, 10)},
            max_total_bits=24,
        )
        assert spec.total_bits <= 24

    def test_alignment(self) -> None:
        """Byte alignment should round up data widths."""
        spec = solve_precision(
            bounds={"v": (-1, 1)},
            min_resolution={"v": 0.01},
            align_to=8,
        )
        assert spec.get("v").data_width % 8 == 0

    def test_single_variable(self) -> None:
        """Should work with a single variable."""
        spec = solve_precision(
            bounds={"x": (0, 255)},
            min_resolution={"x": 0.1},
        )
        assert spec.get("x").can_represent(255)

    def test_mixed_ranges(self) -> None:
        """Variables with very different ranges get different widths."""
        spec = solve_precision(
            bounds={"v": (-32768, 32767), "flag": (0, 1)},
            min_resolution={"v": 0.01, "flag": 0.5},
        )
        assert spec.get("v").data_width > spec.get("flag").data_width


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
