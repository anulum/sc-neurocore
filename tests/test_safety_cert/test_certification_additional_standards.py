# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAdditionalStandards from former test_certification.py

"""Focused suite: TestAdditionalStandards from former test_certification.py."""

from __future__ import annotations

from tests.test_safety_cert.certification_support import *  # noqa: F403

class TestAdditionalStandards:
    def _props(self) -> list[FormalProperty]:
        return [FormalProperty("P1", "sc_lif_neuron", "No overflow", "assert", "proven")]

    def test_generate_do254(self) -> None:
        gen = CertificationGenerator()
        pkg = gen.generate(SafetyStandard.DO_254, SILLevel.SIL_2, ["sc_lif_neuron"], self._props())
        assert len(pkg.checklist) == 6
        assert pkg.standard == SafetyStandard.DO_254

    def test_generate_en50129(self) -> None:
        gen = CertificationGenerator()
        pkg = gen.generate(
            SafetyStandard.EN_50129, SILLevel.SIL_3, ["sc_lif_neuron"], self._props()
        )
        assert len(pkg.checklist) == 6
        assert pkg.standard == SafetyStandard.EN_50129
