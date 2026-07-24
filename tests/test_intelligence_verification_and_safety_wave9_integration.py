# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWave9Integration from former test_intelligence_verification_and_safety.py

"""Focused suite: TestWave9Integration from former test_intelligence_verification_and_safety.py."""

from __future__ import annotations

from tests.intelligence_verification_and_safety_support import *  # noqa: F403


class TestWave9Integration:
    def test_toml_to_report(self, tmp_path):
        from sc_neurocore.compiler.intelligence import (
            load_profiles_from_toml,
            generate_compilation_report,
        )

        toml = tmp_path / "e2e.toml"
        toml.write_text(
            "[[profile]]\n"
            'name = "e2e_custom"\n'
            'vendor = "E2EVendor"\n'
            'platform_class = "custom"\n'
            "data_width = 16\n"
            "fraction = 8\n"
        )
        load_profiles_from_toml(str(toml))
        md = generate_compilation_report("sc_lif", {"v": "a"}, "e2e_custom")
        assert "E2EVendor" in md

    def test_cdc_then_floorplan(self):
        from sc_neurocore.compiler.intelligence import (
            analyze_cdc,
            plan_multi_die_floorplan,
        )

        r = analyze_cdc({"v": "u", "u": "v"}, clock_domains={"v": "clk_a", "u": "clk_b"})
        assert r.total_crossings >= 2
        fp = plan_multi_die_floorplan({"region_a": 500, "region_b": 500})
        assert fp.total_dies >= 1
