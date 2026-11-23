# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMuMax3ScriptGenerator from former test_spintronic_mapper.py

"""Focused suite: TestMuMax3ScriptGenerator from former test_spintronic_mapper.py."""

from __future__ import annotations

from sc_neurocore.spintronic.spintronic_mapper import (
    MuMax3ScriptGenerator,
    SpintronicDeviceConfig,
    SpintronicTech,
)


class TestMuMax3ScriptGenerator:
    """Generating MuMax3 simulation scripts."""

    def test_switching_script(self) -> None:
        """The switching script sets the material and runs."""
        dev = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        script = MuMax3ScriptGenerator.generate_switching(dev)
        assert "Msat" in script
        assert "Aex" in script
        assert "Run(" in script

    def test_skyrmion_script(self) -> None:
        """The skyrmion script places a skyrmion, relaxes and sets DMI."""
        dev = SpintronicDeviceConfig.from_tech(SpintronicTech.SKYRMION)
        script = MuMax3ScriptGenerator.generate_skyrmion(dev)
        assert "Skyrmion" in script
        assert "Relax" in script
        assert "Dind" in script
