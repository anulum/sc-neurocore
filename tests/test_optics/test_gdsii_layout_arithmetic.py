# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLayoutArithmetic from former test_gdsii.py

"""Focused suite: TestLayoutArithmetic from former test_gdsii.py."""

from __future__ import annotations

from gdsii_support import *  # noqa: F403


class TestLayoutArithmetic:
    def test_returns_full_layout_dict(
        self, populated_result: CompilationResult, tmp_path: Path
    ) -> None:
        """Return the emitted file path and geometric parameters for audit."""
        out_path = tmp_path / "cascade.gds"
        info = populated_result.to_gdsii(str(out_path), mzi_length_um=12.5, pitch_um=80.0)
        assert info["filename"] == str(out_path)
        assert info["n_modulators"] == populated_result.num_modulators
        assert info["mzi_length_um"] == pytest.approx(12.5)
        assert info["pitch_um"] == pytest.approx(80.0)
        assert info["target"] == populated_result.target
        # Four MZIs at 80 µm pitch ⇒ final origin at 4·80 = 320 µm.
        assert info["total_length_um"] == pytest.approx(4 * 80.0)

    def test_pitch_scales_total_length_linearly(
        self, populated_result: CompilationResult, tmp_path: Path
    ) -> None:
        """Scale reported cascade length linearly with MZI pitch."""
        info_a = populated_result.to_gdsii(str(tmp_path / "a.gds"), pitch_um=50.0)
        info_b = populated_result.to_gdsii(str(tmp_path / "b.gds"), pitch_um=200.0)
        assert info_b["total_length_um"] == pytest.approx(info_a["total_length_um"] * 4.0)
