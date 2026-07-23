# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEmptyLayoutGuard from former test_gdsii.py

"""Focused suite: TestEmptyLayoutGuard from former test_gdsii.py."""

from __future__ import annotations

from gdsii_support import *  # noqa: F403

class TestEmptyLayoutGuard:
    def test_zero_modulators_raises(self, tmp_path: Path) -> None:
        """Reject an empty physical layout instead of writing a silent shell."""
        empty = CompilationResult(
            target="x",
            num_modulators=0,
            optical_power_mean_mw=0.0,
            phase_coverage_rad=0.0,
            netlist="",
        )
        with pytest.raises(NotImplementedError, match="num_modulators > 0"):
            empty.to_gdsii(str(tmp_path / "empty.gds"))
