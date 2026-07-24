# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCreateNeuron from former test_neuroml_import.py

"""Focused suite: TestCreateNeuron from former test_neuroml_import.py."""

from __future__ import annotations

from tests.neuroml_import_support import *  # noqa: F403


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
