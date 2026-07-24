# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestOCVConfig from former test_signoff.py

"""Focused suite: TestOCVConfig from former test_signoff.py."""

from __future__ import annotations

from tests.test_asic_flow.signoff_support import *  # noqa: F403


class TestOCVConfig:
    def test_default(self) -> None:
        ocv = OCVConfig()
        frag = ocv.generate_sdc_fragment()
        assert "set_timing_derate" in frag
        assert "0.950" in frag

    def test_conservative(self) -> None:
        ocv = OCVConfig.conservative()
        assert ocv.data_cell_early < 0.95
        assert ocv.data_cell_late > 1.05
