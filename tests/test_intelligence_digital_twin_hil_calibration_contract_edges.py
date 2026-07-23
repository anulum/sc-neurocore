# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHILCalibrationContractEdges from former test_intelligence_digital_twin.py

"""Focused suite: TestHILCalibrationContractEdges from former test_intelligence_digital_twin.py."""

from __future__ import annotations

from tests.intelligence_digital_twin_support import *  # noqa: F403

class TestHILCalibrationContractEdges:
    """The remaining validation branches of the HIL contract: each guards a
    distinct malformed sweep specification that the happy-path tests never reach."""

    def test_negative_settle_cycles_rejected(self):
        from sc_neurocore.compiler.intelligence import generate_hil_calibration

        with pytest.raises(ValueError, match="settle_cycles must be >= 0"):
            generate_hil_calibration("sc_lif", {"v": "expr"}, settle_cycles=-1)

    def test_parameter_with_wrong_bound_count_rejected(self):
        from sc_neurocore.compiler.intelligence import generate_hil_calibration

        with pytest.raises(ValueError, match="exactly two bounds"):
            generate_hil_calibration(
                "sc_lif",
                {"v": "expr"},
                parameters={"tau": (0.0, 1.0, 2.0)},
            )

    def test_non_finite_parameter_bounds_rejected(self):
        from sc_neurocore.compiler.intelligence import generate_hil_calibration

        with pytest.raises(ValueError, match="bounds must be finite"):
            generate_hil_calibration(
                "sc_lif",
                {"v": "expr"},
                parameters={"tau": (0.0, float("inf"))},
            )

    def test_empty_parameter_map_rejected(self):
        from sc_neurocore.compiler.intelligence import generate_hil_calibration

        # equations is non-empty (so the earlier guard passes) but the explicit
        # parameter map is empty, leaving no sweep range to calibrate.
        with pytest.raises(ValueError, match="at least one sweep range"):
            generate_hil_calibration("sc_lif", {"v": "expr"}, parameters={})

    def test_observable_absent_from_equations_rejected(self):
        from sc_neurocore.compiler.intelligence import generate_hil_calibration

        with pytest.raises(ValueError, match="not present in equations"):
            generate_hil_calibration(
                "sc_lif",
                {"v": "expr"},
                observables=("v", "phantom"),
            )

    def test_coprime_stride_advances_past_shared_factor(self):
        from sc_neurocore.compiler.intelligence import generate_hil_calibration

        # The second sweep dimension seeds its Latin-hypercube stride at 3; with
        # nine sample points gcd(3, 9) == 3, so the stride search must step past
        # the shared factor before the design matrix can be built.
        r = generate_hil_calibration(
            "sc_lif",
            {"a": "expr", "b": "expr"},
            parameters={"a": (0.0, 1.0), "b": (0.0, 1.0)},
            sample_points=9,
        )
        assert len(r.design_matrix) == 9
        assert r.num_parameters == 2
