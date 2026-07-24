# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSyntheticProfileBoundary from former test_failure_analysis.py

"""Focused suite: TestSyntheticProfileBoundary from former test_failure_analysis.py."""

from __future__ import annotations

from tests.test_safety_cert.failure_analysis_support import *  # noqa: F403


class TestSyntheticProfileBoundary:
    def test_profile_requires_explicit_boolean_acknowledgement(self) -> None:
        fmeda = FMEDA()
        with pytest.raises(ValueError, match="must be a boolean"):
            fmeda.add_sc_standard_modes(
                "neuron",
                acknowledge_synthetic_profile=_unsafe("yes"),
            )
        with pytest.raises(ValueError, match="requires acknowledge"):
            fmeda.add_sc_standard_modes("neuron")

    def test_empty_fmeda_report_is_explicitly_unassessed(self) -> None:
        assert "Status: not assessed" in FMEDA().generate_report()
