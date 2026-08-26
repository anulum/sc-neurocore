# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGDSFileRoundtrip from former test_gdsii.py

"""Focused suite: TestGDSFileRoundtrip from former test_gdsii.py."""

from __future__ import annotations

import pytest


pytest.importorskip(
    "gdsfactory",
    reason="gdsfactory is an optional dep (install via `pip install sc-neurocore[optics]`)",
)

from gdsii_support import *  # noqa: E402,F403


class TestGDSFileRoundtrip:
    def test_file_created_and_nonempty(
        self, populated_result: CompilationResult, tmp_path: Path
    ) -> None:
        """Write a non-empty GDSII file for a populated cascade."""
        out_path = tmp_path / "roundtrip.gds"
        populated_result.to_gdsii(str(out_path))
        assert out_path.exists()
        size = out_path.stat().st_size
        assert size > 0

    def test_file_reads_back_via_gdsfactory(
        self, populated_result: CompilationResult, tmp_path: Path
    ) -> None:
        """Parse the exported GDSII file back through gdsfactory."""
        out_path = tmp_path / "parse.gds"
        populated_result.to_gdsii(str(out_path))
        # gdsfactory's low-level reader (via gdspy / klayout) must parse it.
        comp = gf.read.import_gds(str(out_path))
        assert comp is not None
