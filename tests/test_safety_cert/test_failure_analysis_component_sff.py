# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestComponentSFF from former test_failure_analysis.py

"""Focused suite: TestComponentSFF from former test_failure_analysis.py."""

from __future__ import annotations

from tests.test_safety_cert.failure_analysis_support import *  # noqa: F403


class TestComponentSFF:
    def test_sff_by_component(self) -> None:
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("neuron", acknowledge_synthetic_profile=True)
        fmeda.add_sc_standard_modes("encoder", acknowledge_synthetic_profile=True)
        sff_map = fmeda.sff_by_component()
        assert "neuron" in sff_map
        assert "encoder" in sff_map
        assert 0 < sff_map["neuron"] <= 1.0

    def test_sff_single_component(self) -> None:
        fmeda = FMEDA()
        fmeda.add_sc_standard_modes("lif", acknowledge_synthetic_profile=True)
        sff_map = fmeda.sff_by_component()
        assert len(sff_map) == 1
