# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDiagnoseHardware from former test_doctor.py

"""Focused suite: TestDiagnoseHardware from former test_doctor.py."""

from __future__ import annotations

from sc_neurocore.doctor import (
    diagnose,
)
from sc_neurocore.doctor.diagnose import (
    Severity,
)


class TestDiagnoseHardware:
    """Diagnosing the hardware fit."""

    def test_small_network_fits(self) -> None:
        """A small network gets a fit finding."""
        r = diagnose([(4, 2)], target="artix7")
        hw = [f for f in r.findings if f.category.startswith("hardware")]
        assert len(hw) >= 1

    def test_overprovisioned(self) -> None:
        """An oversized device is reported."""
        r = diagnose([(4, 2)], target="artix7")
        over = [f for f in r.findings if f.category == "hardware_overprovisioned"]
        assert len(over) >= 1

    def test_large_network_exceeds(self) -> None:
        """A network too large for the device is critical."""
        r = diagnose([(256, 128), (128, 64)], target="ice40", bitstream_length=512)
        hw = [f for f in r.findings if f.category == "hardware_fit"]
        assert any(f.severity == Severity.CRITICAL for f in hw)
