# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestImportIAFCell from former test_neuroml_import.py

"""Focused suite: TestImportIAFCell from former test_neuroml_import.py."""

from __future__ import annotations

from pathlib import Path
import pytest
from sc_neurocore.adapters.neuroml import (
    import_neuroml,
)
from tests.neuroml_import_support import (
    _write_nml,
)
from textwrap import (
    dedent,
)
from tests.neuroml_import_support import ensure_fixtures as ensure_fixtures


class TestImportIAFCell:
    """Importing the integrate-and-fire cell variants."""

    def test_basic_lif(self, tmp_path: Path) -> None:
        """An iafCell imports as a LIF cell with its parameters in model units."""
        f = _write_nml(
            tmp_path / "lif.nml",
            dedent("""\
            <iafCell id="lif0" C="100pF" leakConductance="10nS"
                     leakReversal="-65mV" thresh="-55mV" reset="-70mV"/>
        """),
        )
        cells = import_neuroml(f)
        assert len(cells) == 1
        c = cells[0]
        assert c.cell_id == "lif0"
        assert c.cell_type == "StochasticLIFNeuron"
        assert c.params["tau_mem"] == pytest.approx(10.0, rel=0.01)
        assert c.params["v_threshold"] == pytest.approx(10.0, rel=0.01)

    def test_ref_cell(self, tmp_path: Path) -> None:
        """An iafRefCell imports with its refractory period."""
        f = _write_nml(
            tmp_path / "ref.nml",
            dedent("""\
            <iafRefCell id="lif_ref" C="200pF" leakConductance="20nS"
                        leakReversal="-60mV" thresh="-50mV" reset="-65mV"
                        refract="5ms"/>
        """),
        )
        cells = import_neuroml(f)
        assert cells[0].params["refractory_period"] == 5

    def test_tau_cell(self, tmp_path: Path) -> None:
        """An iafTauCell imports with its time constant."""
        f = _write_nml(
            tmp_path / "tau.nml",
            dedent("""\
            <iafTauCell id="tau0" tau="20ms" leakReversal="-65mV"
                        thresh="-55mV" reset="-70mV"/>
        """),
        )
        cells = import_neuroml(f)
        assert cells[0].params["tau_mem"] == 20.0

    def test_tau_ref_cell_carries_refractory_period(self, tmp_path: Path) -> None:
        """An iafTauRefCell imports with its time constant and refractory period."""
        f = _write_nml(
            tmp_path / "tau_ref.nml",
            dedent("""\
            <iafTauRefCell id="tau_ref0" tau="20ms" leakReversal="-65mV"
                           thresh="-55mV" reset="-70mV" refract="3ms"/>
        """),
        )
        cells = import_neuroml(f)
        assert cells[0].params["tau_mem"] == 20.0
        assert cells[0].params["refractory_period"] == 3
