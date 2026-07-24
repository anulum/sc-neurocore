# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDataTypes from former test_photonic_noc.py

"""Focused suite: TestDataTypes from former test_photonic_noc.py."""

from __future__ import annotations

from photonic_noc_support import *  # noqa: F403


class TestDataTypes:
    """Photonic data class tests."""

    def test_waveguide_type_enum(self) -> None:
        assert WaveguideType.STRIP.value == "strip"
        assert WaveguideType.RIB.value == "rib"

    def test_waveguide_segment(self) -> None:
        wg = WaveguideSegment(source=0, target=1, length_um=500.0, loss_db=1.0)
        assert wg.source == 0
        assert wg.length_um == 500.0

    def test_mzi_gate(self) -> None:
        mzi = MZIGate(gate_id="mzi_0", operation="MUL", phase_shift_rad=math.pi / 2)
        assert mzi.operation == "MUL"
        assert abs(mzi.phase_shift_rad - math.pi / 2) < 1e-10

    def test_wdm_channel(self) -> None:
        ch = WDMChannel(channel_id=0, wavelength_nm=1550.0, signal_name="a")
        assert ch.wavelength_nm == 1550.0
