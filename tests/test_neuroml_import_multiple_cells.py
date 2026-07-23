# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMultipleCells from former test_neuroml_import.py

"""Focused suite: TestMultipleCells from former test_neuroml_import.py."""

from __future__ import annotations

from tests.neuroml_import_support import *  # noqa: F403

class TestMultipleCells:
    def test_mixed_file(self, tmp_path):
        f = _write_nml(
            tmp_path / "mix.nml",
            dedent("""\
            <iafCell id="lif1" C="100pF" leakConductance="10nS"
                     leakReversal="-65mV" thresh="-55mV" reset="-70mV"/>
            <izhikevichCell id="izh1" v0="-65mV" thresh="30mV"
                            a="0.02" b="0.2" c="-65" d="8"/>
            <adExIaFCell id="adex1" C="281pF" gL="30nS" EL="-70.6mV"
                         VT="-50.4mV" thresh="-40mV" reset="-70.6mV"
                         delT="2mV" tauw="144ms" a="4nS" b="0.0805nA"
                         refract="0ms"/>
        """),
        )
        cells = import_neuroml(f)
        assert len(cells) == 3
        types = {c.cell_type for c in cells}
        assert types == {"StochasticLIFNeuron", "SCIzhikevichNeuron", "AdExNeuron"}
