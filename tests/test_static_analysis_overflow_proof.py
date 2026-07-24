# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestOverflowProof from former test_static_analysis.py

"""Focused suite: TestOverflowProof from former test_static_analysis.py."""

from __future__ import annotations

from tests.static_analysis_support import *  # noqa: F403


class TestOverflowProof:
    """Test formal overflow proofs via interval arithmetic."""

    def test_lif_safe_at_q88(self) -> None:
        """LIF derivative should be provably safe at Q8.8 with bounded inputs."""
        result = prove_no_overflow(
            "-(v - v_rest) / tau_m + R * I / C",
            bounds={
                "v": (-128, 127),
                "v_rest": (-65, -65),
                "tau_m": (10, 10),
                "R": (1, 1),
                "I": (0, 100),
                "C": (1, 1),
            },
            data_width=16,
            fraction=8,
        )
        assert result.proven_safe, (
            f"LIF should be safe at Q8.8: "
            f"result=[{result.expr_interval.lo:.1f}, {result.expr_interval.hi:.1f}]"
        )

    def test_overflow_detected(self) -> None:
        """Detect overflow when values exceed Q1.7 range."""
        result = prove_no_overflow(
            "v + I",
            bounds={"v": (-65, 30), "I": (0, 100)},
            data_width=8,
            fraction=7,
        )
        assert not result.proven_safe

    def test_safe_normalised_model(self) -> None:
        """Normalised FHN model within operating range should be safe at Q4.12."""
        result = prove_no_overflow(
            "a * (v - v * v * v) - w + I",
            bounds={
                "a": (0.5, 0.5),
                "v": (-1.5, 1.5),
                "w": (-1.5, 1.5),
                "I": (0, 0.5),
            },
            data_width=16,
            fraction=12,
        )
        assert result.proven_safe

    def test_margin_values(self) -> None:
        """Margins should be positive when safe, negative when unsafe."""
        safe = prove_no_overflow(
            "a + b",
            bounds={"a": (0, 10), "b": (0, 10)},
            data_width=16,
            fraction=8,
        )
        assert safe.margin_lo > 0
        assert safe.margin_hi > 0

    def test_unsigned_format(self) -> None:
        """Unsigned format has min=0, larger max."""
        result = prove_no_overflow(
            "a + b",
            bounds={"a": (0, 100), "b": (0, 100)},
            data_width=16,
            fraction=8,
            signed=False,
        )
        assert result.q_min == 0.0
        assert result.proven_safe
