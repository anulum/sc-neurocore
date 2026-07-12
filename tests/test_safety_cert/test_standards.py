# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Safety Certification Generator Tests

"""Focused tests for standards."""

from sc_neurocore.safety_cert.safety_cert import (
    SIL_TO_ASIL,
    ASILLevel,
    SILLevel,
)


class TestSILASIL:
    def test_sil_to_asil(self) -> None:
        assert SIL_TO_ASIL[SILLevel.SIL_1] == ASILLevel.ASIL_A
        assert SIL_TO_ASIL[SILLevel.SIL_4] == ASILLevel.ASIL_D
