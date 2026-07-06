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
    ImportedCell,
    _parse_current_pa,
    _parse_unit_value,
    create_neuron,
    import_neuroml,
)
from sc_neurocore.neurons.models import Izhikevich2007Neuron
from sc_neurocore.neurons.models.adex import AdExNeuron

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

    def test_tau_ref_cell_carries_refractory_period(self, tmp_path):
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

    def test_izhikevich2007_instantiation_preserves_biophysical_state(self, tmp_path):
        f = _write_nml(
            tmp_path / "izh07.nml",
            dedent("""\
            <izhikevich2007Cell id="izh07" C="100pF" k="0.7nS_per_mV"
                                vr="-60mV" vt="-40mV" vpeak="35mV"
                                a="0.03per_ms" b="-2nS" c="-50mV" d="100pA"
                                v0="-60mV"/>
        """),
        )
        cells = import_neuroml(f)

        neuron = create_neuron(cells[0])

        assert isinstance(neuron, Izhikevich2007Neuron)
        assert neuron.get_state() == {"v": pytest.approx(-60.0), "u": pytest.approx(0.0)}

    def test_adex_instantiation_adapts(self, tmp_path):
        # Regression: the imported AdEx parameters must map onto AdExNeuron's own
        # constructor names/units (previously they used NeuroML names and crashed
        # with TypeError), and the pA-unit spike-triggered adaptation must produce
        # a lengthening inter-spike interval under a sustained supra-rheobase drive.
        f = _write_nml(
            tmp_path / "adex.nml",
            dedent("""\
            <adExIaFCell id="adex0" C="281pF" gL="30nS" EL="-70.6mV"
                         VT="-50.4mV" thresh="-40mV" reset="-70.6mV"
                         delT="2mV" tauw="144ms" a="4nS" b="0.0805nA"/>
        """),
        )
        cells = import_neuroml(f)
        neuron = create_neuron(cells[0])
        assert isinstance(neuron, AdExNeuron)

        spikes = [neuron.step(1000.0) for _ in range(3000)]
        fire_idx = [i for i, s in enumerate(spikes) if s]
        assert len(fire_idx) >= 4
        intervals = [fire_idx[k + 1] - fire_idx[k] for k in range(len(fire_idx) - 1)]
        # Spike-frequency adaptation: later intervals exceed the first.
        assert intervals[-1] > intervals[0]

    def test_unknown_cell_type_raises(self):
        bogus = ImportedCell(
            cell_id="mystery",
            cell_type="NotARealNeuron",
            params={},
            source_tag="mysteryCell",
        )
        with pytest.raises(ValueError, match="Unknown cell type: NotARealNeuron"):
            create_neuron(bogus)


class TestParseHelpers:
    def test_parse_unit_value_none_is_zero(self):
        assert _parse_unit_value(None) == 0.0

    def test_parse_unit_value_dimensionless_falls_through(self):
        # No recognised unit suffix -> parsed as a bare float.
        assert _parse_unit_value("0.7") == pytest.approx(0.7)

    def test_parse_current_pa_none_is_zero(self):
        assert _parse_current_pa(None) == 0.0

    def test_parse_current_pa_dimensionless_falls_through(self):
        assert _parse_current_pa("42") == pytest.approx(42.0)


class TestIzhikevich2007Neuron:
    def test_euler_step_matches_biophysical_equations_below_threshold(self):
        neuron = Izhikevich2007Neuron(
            C=100.0,
            k=0.7,
            vr=-60.0,
            vt=-40.0,
            vpeak=35.0,
            a=0.03,
            b=-2.0,
            c=-50.0,
            d=100.0,
            v0=-61.0,
            dt=0.1,
            integrator="euler",
        )

        spike = neuron.step(70.0)

        expected_dv = (0.7 * (-1.0) * (-21.0) - 2.0 + 70.0) / 100.0
        expected_du = 0.03 * (-2.0 * (-1.0) - 2.0)
        assert spike == 0
        assert neuron.v == pytest.approx(-61.0 + 0.1 * expected_dv)
        assert neuron.u == pytest.approx(2.0 + 0.1 * expected_du)

    def test_spike_reset_uses_vpeak_c_and_d(self):
        neuron = Izhikevich2007Neuron(
            C=100.0,
            k=0.7,
            vr=-60.0,
            vt=-40.0,
            vpeak=35.0,
            a=0.03,
            b=-2.0,
            c=-50.0,
            d=100.0,
            v0=34.0,
            dt=1.0,
            integrator="euler",
        )

        spike = neuron.step(500.0)

        assert spike == 1
        assert neuron.v == pytest.approx(-50.0)
        assert neuron.u == pytest.approx(-88.0)
