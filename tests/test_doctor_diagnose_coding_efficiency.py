# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDiagnoseCodingEfficiency from former test_doctor.py

"""Focused suite: TestDiagnoseCodingEfficiency from former test_doctor.py."""

from __future__ import annotations

from tests.doctor_support import *  # noqa: F403

class TestDiagnoseCodingEfficiency:
    def test_overprovisioned_coding(self):
        r = diagnose([(10, 8)], target="artix7", bitstream_length=512)
        cod = [f for f in r.findings if f.category == "coding_overprovisioned"]
        assert len(cod) >= 1

    def test_underprovisioned_coding(self):
        layers = [(64, 64), (64, 64), (64, 64), (64, 32)]
        r = diagnose(layers, target="artix7", bitstream_length=32)
        cod = [f for f in r.findings if f.category == "coding_underprovisioned"]
        assert len(cod) >= 1
