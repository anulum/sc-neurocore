# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestImportAdEx from former test_neuroml_import.py

"""Focused suite: TestImportAdEx from former test_neuroml_import.py."""

from __future__ import annotations

from tests.neuroml_import_support import *  # noqa: F403

class TestImportAdEx:
    def test_adex(self, tmp_path):
        f = _write_nml(
            tmp_path / "adex.nml",
            dedent("""\
            <adExIaFCell id="adex0" C="281pF" gL="30nS" EL="-70.6mV"
                         VT="-50.4mV" thresh="-40mV" reset="-70.6mV"
                         delT="2mV" tauw="144ms" a="4nS" b="0.0805nA"
                         refract="0ms"/>
        """),
        )
        cells = import_neuroml(f)
        c = cells[0]
        assert c.cell_type == "AdExNeuron"
        # Attributes map onto AdExNeuron's own constructor names / units.
        assert c.params["c_m"] == pytest.approx(281.0, rel=0.01)
        assert c.params["tau_w"] == pytest.approx(144.0, rel=0.01)
        # tau = C / g_L (pF / nS = ms).
        assert c.params["tau"] == pytest.approx(281.0 / 30.0, rel=0.01)
        assert c.params["v_rest"] == pytest.approx(-70.6, rel=0.01)
        assert c.params["v_rh"] == pytest.approx(-50.4, rel=0.01)
        # b is a spike-triggered adaptation *current*: 0.0805 nA = 80.5 pA.
        assert c.params["b"] == pytest.approx(80.5, rel=0.01)
