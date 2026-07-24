# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestImportIzhikevich from former test_neuroml_import.py

"""Focused suite: TestImportIzhikevich from former test_neuroml_import.py."""

from __future__ import annotations

from tests.neuroml_import_support import *  # noqa: F403


class TestImportIzhikevich:
    def test_2003_dimensionless(self, tmp_path):
        f = _write_nml(
            tmp_path / "izh.nml",
            dedent("""\
            <izhikevichCell id="izh_rs" v0="-65mV" thresh="30mV"
                            a="0.02" b="0.2" c="-65" d="8"/>
        """),
        )
        cells = import_neuroml(f)
        c = cells[0]
        assert c.cell_type == "SCIzhikevichNeuron"
        assert c.params["a"] == 0.02
        assert c.params["c"] == -65.0

    def test_2007_biophysical(self, tmp_path):
        f = _write_nml(
            tmp_path / "izh07.nml",
            dedent("""\
            <izhikevich2007Cell id="izh07" C="100pF" k="0.7nS_per_mV"
                                vr="-60mV" vt="-40mV" vpeak="35mV"
                                a="0.03per_ms" b="-2nS" c="-50mV" d="100pA"
                                v0="-61mV"/>
        """),
        )
        cells = import_neuroml(f)
        cell = cells[0]
        assert cell.cell_type == "Izhikevich2007Neuron"
        assert cell.params["C"] == pytest.approx(100.0)
        assert cell.params["k"] == pytest.approx(0.7)
        assert cell.params["vr"] == pytest.approx(-60.0)
        assert cell.params["vt"] == pytest.approx(-40.0)
        assert cell.params["vpeak"] == pytest.approx(35.0)
        assert cell.params["a"] == pytest.approx(0.03)
        assert cell.params["b"] == pytest.approx(-2.0)
        assert cell.params["c"] == pytest.approx(-50.0)
        assert cell.params["d"] == pytest.approx(100.0)
        assert cell.params["v0"] == pytest.approx(-61.0)
        assert cell.params["integrator"] == "rk4"
        assert "_neuroml2007_raw" not in cell.params
