# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPowerDomainValidation from former test_chiplet_gen_edge_cases.py

"""Focused suite: TestPowerDomainValidation from former test_chiplet_gen_edge_cases.py."""

from __future__ import annotations

from chiplet_gen_edge_cases_support import *  # noqa: F403

class TestPowerDomainValidation:
    def test_duplicate_die_ids_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            PowerDomain(domain_id=0, die_ids=[1, 1], voltage_mv=800)
