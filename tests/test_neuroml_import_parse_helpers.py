# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - NeuroML attribute strictness and mapping notes

"""The NeuroML importer refuses what it would otherwise invent, and says what it maps."""

from __future__ import annotations

from tests.neuroml_import_support import *  # noqa: F403

_IAF = (
    '<iafCell id="lif0" C="100pF" leakConductance="10nS" '
    'leakReversal="-65mV" thresh="-55mV" reset="-70mV"/>'
)


class TestAttributeStrictness:
    @pytest.mark.parametrize(
        ("attributes", "message"),
        [
            (
                'C="100pF" leakReversal="-65mV" thresh="-55mV" reset="-70mV"',
                "'leakConductance' is missing",
            ),
            (
                'C="100" leakConductance="10nS" leakReversal="-65mV" thresh="-55mV" reset="-70mV"',
                "needs a capacitance unit",
            ),
            (
                'C="100mV" leakConductance="10nS" leakReversal="-65mV" thresh="-55mV" reset="-70mV"',
                "needs a capacitance unit",
            ),
            (
                'C="many" leakConductance="10nS" leakReversal="-65mV" thresh="-55mV" reset="-70mV"',
                "is not a number with a unit",
            ),
            (
                'C="1e999pF" leakConductance="10nS" leakReversal="-65mV" thresh="-55mV" reset="-70mV"',
                "is not finite",
            ),
            (
                'C="0pF" leakConductance="10nS" leakReversal="-65mV" thresh="-55mV" reset="-70mV"',
                "needs positive C",
            ),
        ],
        ids=[
            "missing",
            "bare-number",
            "wrong-dimension",
            "not-a-number",
            "infinite",
            "zero-capacitance",
        ],
    )
    def test_an_iaf_attribute_that_cannot_be_read_as_stated_is_refused(
        self, tmp_path: Path, attributes: str, message: str
    ) -> None:
        f = _write_nml(tmp_path / "bad.nml", f'<iafCell id="lif0" {attributes}/>', valid=False)
        with pytest.raises(ValueError, match=message):
            import_neuroml(f)

    def test_a_cell_without_an_id_is_refused(self, tmp_path: Path) -> None:
        f = _write_nml(
            tmp_path / "anon.nml",
            '<iafTauCell tau="20ms" leakReversal="-65mV" thresh="-55mV" reset="-70mV"/>',
            valid=False,
        )
        with pytest.raises(ValueError, match="<iafTauCell> has no id"):
            import_neuroml(f)

    def test_a_dimensionless_attribute_with_a_unit_is_refused(self, tmp_path: Path) -> None:
        f = _write_nml(
            tmp_path / "izh.nml",
            '<izhikevichCell id="izh" v0="-65mV" thresh="30mV" a="0.02ms" b="0.2" c="-65" d="8"/>',
            valid=False,
        )
        with pytest.raises(ValueError, match="'a' is dimensionless but has unit 'ms'"):
            import_neuroml(f)

    def test_an_adex_cell_needs_positive_capacitance_and_conductance(self, tmp_path: Path) -> None:
        f = _write_nml(
            tmp_path / "adex.nml",
            '<adExIaFCell id="adex0" C="281pF" gL="0nS" EL="-70.6mV" VT="-50.4mV" '
            'thresh="-40mV" reset="-70.6mV" delT="2mV" tauw="144ms" a="4nS" '
            'b="0.0805nA" refract="0ms"/>',
        )
        with pytest.raises(ValueError, match="needs positive C and gL"):
            import_neuroml(f)


class TestDocumentStrictness:
    def test_elements_the_importer_does_not_model_are_refused(self, tmp_path: Path) -> None:
        f = _write_nml(
            tmp_path / "net.nml",
            _IAF
            + '<pulseGenerator id="pg" delay="10ms" duration="100ms" amplitude="0.1nA"/>'
            + '<network id="net"><population id="pop" component="lif0" size="2"/></network>',
        )
        with pytest.raises(
            ValueError, match="does not model: network, pulseGenerator; it imports point-cell"
        ):
            import_neuroml(f)

    def test_documentation_elements_are_ignored(self, tmp_path: Path) -> None:
        f = _write_nml(tmp_path / "doc.nml", "<notes>A single cell.</notes>" + _IAF)
        assert [cell.cell_id for cell in import_neuroml(f)] == ["lif0"]

    def test_the_root_must_be_a_neuroml_document(self, tmp_path: Path) -> None:
        f = tmp_path / "other.xml"
        f.write_text("<lems/>", encoding="utf-8")
        with pytest.raises(ValueError, match="root must be <neuroml>, got <lems>"):
            import_neuroml(f)


class TestMappingNotes:
    def test_an_iaf_cell_names_its_timestep_voltage_shift_and_resistance(
        self, tmp_path: Path
    ) -> None:
        (cell,) = import_neuroml(_write_nml(tmp_path / "lif.nml", _IAF))
        assert cell.notes == (
            "NeuroML cells carry no timestep; the model runs at dt=1.0 ms",
            "voltages are relative to the leak reversal -65.0 mV, which becomes 0 in the model",
            "input current enters through a normalised resistance of 1; the document "
            "implies 1/leakConductance = 100.0 MOhm",
        )

    def test_a_fractional_refractory_period_is_reported_as_rounded(self, tmp_path: Path) -> None:
        f = _write_nml(
            tmp_path / "ref.nml",
            '<iafTauRefCell id="r" tau="20ms" leakReversal="-65mV" thresh="-55mV" '
            'reset="-70mV" refract="2.6ms"/>',
        )
        (cell,) = import_neuroml(f)
        assert cell.params["refractory_period"] == 3
        assert "refractory period 2.6 ms realised as 3 whole timesteps of 1.0 ms" in cell.notes

    def test_a_2003_izhikevich_cell_names_what_the_model_has_no_place_for(
        self, tmp_path: Path
    ) -> None:
        f = _write_nml(
            tmp_path / "izh.nml",
            '<izhikevichCell id="izh" v0="-70mV" thresh="25mV" a="0.02" b="0.2" c="-65" d="8"/>',
        )
        (cell,) = import_neuroml(f)
        assert any("v0=-70.0 mV is not carried" in note for note in cell.notes)
        assert any("thresh=25.0 mV is not carried" in note for note in cell.notes)

    def test_an_adex_refractory_period_is_reported_only_when_it_is_lost(
        self, tmp_path: Path
    ) -> None:
        body = (
            '<adExIaFCell id="{id}" C="281pF" gL="30nS" EL="-70.6mV" VT="-50.4mV" '
            'thresh="-40mV" reset="-70.6mV" delT="2mV" tauw="144ms" a="4nS" '
            'b="0.0805nA" refract="{refract}"/>'
        )
        f = _write_nml(
            tmp_path / "adex.nml",
            body.format(id="none", refract="0ms") + body.format(id="some", refract="2ms"),
        )
        without, with_refract = import_neuroml(f)
        assert not any("refractory" in note for note in without.notes)
        assert (
            "refractory period 2.0 ms is not carried; AdExNeuron has no refractory period"
            in with_refract.notes
        )
