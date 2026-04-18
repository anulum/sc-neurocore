# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for NeuroML 2 importer

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from sc_neurocore.adapters.neuroml import (
    create_neuron,
    import_neuroml,
)

FIXTURES = Path(__file__).parent / "fixtures" / "neuroml"


@pytest.fixture(autouse=True)
def ensure_fixtures(tmp_path):
    """Create test NeuroML files in tmp_path."""
    d = tmp_path / "neuroml"
    d.mkdir()
    yield d


def _write_nml(path: Path, body: str) -> Path:
    header = dedent("""\
    <neuroml xmlns="http://www.neuroml.org/schema/neuroml2"
             id="test">
    """)
    path.write_text(header + body + "\n</neuroml>")
    return path


class TestImportIAFCell:
    def test_basic_lif(self, tmp_path):
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

    def test_ref_cell(self, tmp_path):
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

    def test_tau_cell(self, tmp_path):
        f = _write_nml(
            tmp_path / "tau.nml",
            dedent("""\
            <iafTauCell id="tau0" tau="20ms" leakReversal="-65mV"
                        thresh="-55mV" reset="-70mV"/>
        """),
        )
        cells = import_neuroml(f)
        assert cells[0].params["tau_mem"] == 20.0


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
            <izhikevich2007Cell id="izh07" C="100pF" k="0.7"
                                vr="-60mV" vt="-40mV" vpeak="35mV"
                                a="0.03" b="-2" c="-50" d="100"/>
        """),
        )
        cells = import_neuroml(f)
        assert cells[0].cell_type == "SCIzhikevichNeuron"


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
        assert c.params["C"] == pytest.approx(281.0, rel=0.01)
        assert c.params["tau_w"] == pytest.approx(144.0, rel=0.01)


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


class TestCreateNeuron:
    def test_lif_instantiation(self, tmp_path):
        f = _write_nml(
            tmp_path / "lif.nml",
            dedent("""\
            <iafTauCell id="lif" tau="20ms" leakReversal="-65mV"
                        thresh="-55mV" reset="-70mV"/>
        """),
        )
        cells = import_neuroml(f)
        neuron = create_neuron(cells[0])
        spike = neuron.step(100.0)
        assert spike in (0, 1)

    def test_izhikevich_instantiation(self, tmp_path):
        f = _write_nml(
            tmp_path / "izh.nml",
            dedent("""\
            <izhikevichCell id="izh" v0="-65mV" thresh="30mV"
                            a="0.02" b="0.2" c="-65" d="8"/>
        """),
        )
        cells = import_neuroml(f)
        neuron = create_neuron(cells[0])
        spikes = sum(neuron.step(10.0) for _ in range(100))
        assert spikes > 0
