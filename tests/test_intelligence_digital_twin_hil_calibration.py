# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHILCalibration from former test_intelligence_digital_twin.py

"""Focused suite: TestHILCalibration from former test_intelligence_digital_twin.py."""

from __future__ import annotations

from tests.intelligence_digital_twin_support import *  # noqa: F403

class TestHILCalibration:
    def test_basic(self):
        from sc_neurocore.compiler.intelligence import generate_hil_calibration

        r = generate_hil_calibration("sc_lif", {"v": "expr", "u": "expr"})
        assert r.num_parameters == 2
        assert len(r.protocol_steps) >= 5

    def test_custom_ranges(self):
        from sc_neurocore.compiler.intelligence import generate_hil_calibration

        r = generate_hil_calibration(
            "sc_lif",
            {"v": "expr"},
            parameters={"tau": (-1.0, 1.0)},
        )
        assert r.sweep_ranges["tau"] == (-1.0, 1.0)

    def test_protocol_contains_design_matrix_and_acceptance_metadata(self):
        from sc_neurocore.compiler.intelligence import generate_hil_calibration

        r = generate_hil_calibration(
            "sc_lif",
            {"v": "-v/tau"},
            parameters={"tau": (5.0, 50.0), "threshold": (0.5, 2.0)},
            sample_points=5,
            repetitions=3,
            settle_cycles=16,
            acceptance_tolerance=1e-3,
            correction_model="weighted_least_squares",
        )

        assert r.sample_count == 15
        assert len(r.design_matrix) == 5
        assert {tuple(point) for point in r.design_matrix} == {("tau", "threshold")}
        assert r.observables == ("v",)
        assert r.correction_model == "weighted_least_squares"
        assert r.acceptance_tolerance == 1e-3
        assert any("settle 16 cycles" in step for step in r.protocol_steps)
        assert any("weighted_least_squares" in step for step in r.protocol_steps)

    def test_rejects_invalid_calibration_contract(self):
        from sc_neurocore.compiler.intelligence import generate_hil_calibration

        invalid_cases = [
            dict(module_name="", equations={"v": "expr"}, parameters={"tau": (0.0, 1.0)}),
            dict(module_name="sc_lif", equations={}, parameters={"tau": (0.0, 1.0)}),
            dict(module_name="sc_lif", equations={"v": "expr"}, parameters={"tau": (1.0, 1.0)}),
            dict(module_name="sc_lif", equations={"v": "expr"}, sample_points=1),
            dict(module_name="sc_lif", equations={"v": "expr"}, repetitions=0),
            dict(module_name="sc_lif", equations={"v": "expr"}, acceptance_tolerance=0.0),
        ]

        for kwargs in invalid_cases:
            with pytest.raises(ValueError):
                generate_hil_calibration(**kwargs)
