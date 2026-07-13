# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bioware data-contract tests

"""Tests for stable biological-interface data contracts."""

from __future__ import annotations

from sc_neurocore.bioware.bioware import MEAConfig, MEALayout


class TestMEAConfig:
    def test_defaults(self) -> None:
        cfg = MEAConfig()
        assert cfg.num_channels == 60
        assert cfg.sample_rate_hz == 20_000.0

    def test_from_layout_60(self) -> None:
        cfg = MEAConfig.from_layout(MEALayout.MEA_60)
        assert cfg.num_channels == 60

    def test_from_layout_4096(self) -> None:
        cfg = MEAConfig.from_layout(MEALayout.MEA_4096)
        assert cfg.num_channels == 4096
        assert cfg.electrode_pitch_um < 20.0

    def test_all_layouts(self) -> None:
        for layout in MEALayout:
            cfg = MEAConfig.from_layout(layout)
            assert cfg.num_channels > 0


# ── SpikeDetector Tests ──────────────────────────────────────────────
