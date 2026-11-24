# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDiagnoseArchitecture from former test_doctor.py

"""Focused suite: TestDiagnoseArchitecture from former test_doctor.py."""

from __future__ import annotations

from sc_neurocore.doctor import (
    diagnose,
)


class TestDiagnoseArchitecture:
    """Diagnosing the network architecture."""

    def test_bottleneck(self) -> None:
        """A bottleneck layer is reported."""
        r = diagnose([(256, 256), (256, 8)], target="artix7")
        bn = [f for f in r.findings if f.category == "architecture_bottleneck"]
        assert len(bn) >= 1

    def test_small_capacity(self) -> None:
        """A network with little capacity is reported."""
        r = diagnose([(64, 4)], target="artix7")
        cap = [f for f in r.findings if f.category == "architecture_capacity"]
        assert len(cap) >= 1
