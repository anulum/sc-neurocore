# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NeuroML documents written by libNeuroML itself

"""Documents the upstream NeuroML library writes are imported value for value."""

from __future__ import annotations

from pathlib import Path

import neuroml
from neuroml.utils import validate_neuroml2
from neuroml.writers import NeuroMLWriter
import pytest

from sc_neurocore.adapters.neuroml import create_neuron, import_neuroml


def _write(document: neuroml.NeuroMLDocument, path: Path) -> Path:
    NeuroMLWriter.write(document, str(path))
    validate_neuroml2(str(path))
    return path


def test_every_supported_cell_written_by_libneuroml_imports_in_the_models_units(
    tmp_path: Path,
) -> None:
    """Units other than the base unit are converted; nothing is defaulted."""
    document = neuroml.NeuroMLDocument(id="upstream")
    document.add(
        neuroml.IafCell,
        id="iaf",
        C="0.2nF",
        leak_conductance="0.02uS",
        leak_reversal="-0.06V",
        thresh="-50mV",
        reset="-65mV",
    )
    document.add(
        neuroml.IafRefCell,
        id="iafref",
        C="100pF",
        leak_conductance="10nS",
        leak_reversal="-65mV",
        thresh="-55mV",
        reset="-70mV",
        refract="0.004s",
    )
    document.add(
        neuroml.IafTauCell,
        id="tau",
        tau="0.02s",
        leak_reversal="-65mV",
        thresh="-55mV",
        reset="-70mV",
    )
    document.add(
        neuroml.IafTauRefCell,
        id="tauref",
        tau="20ms",
        leak_reversal="-65mV",
        thresh="-55mV",
        reset="-70mV",
        refract="3ms",
    )
    document.add(
        neuroml.IzhikevichCell,
        id="izh",
        v0="-65mV",
        thresh="30mV",
        a="0.02",
        b="0.2",
        c="-65",
        d="8",
    )
    document.add(
        neuroml.Izhikevich2007Cell,
        id="izh07",
        C="0.1nF",
        v0="-60mV",
        k="7e-7S_per_V",
        vr="-60mV",
        vt="-40mV",
        vpeak="35mV",
        a="30per_s",
        b="-0.002uS",
        c="-50mV",
        d="0.1nA",
    )
    document.add(
        neuroml.AdExIaFCell,
        id="adex",
        C="281pF",
        g_l="30nS",
        EL="-70.6mV",
        reset="-70.6mV",
        VT="-50.4mV",
        thresh="-40mV",
        del_t="2mV",
        tauw="0.144s",
        refract="0ms",
        a="4nS",
        b="80.5pA",
    )

    cells = {
        cell.cell_id: cell for cell in import_neuroml(_write(document, tmp_path / "cells.nml"))
    }

    # libNeuroML writes cells grouped by type, in its own order.
    assert set(cells) == {"iaf", "iafref", "tau", "tauref", "izh", "izh07", "adex"}
    iaf = cells["iaf"].params
    assert iaf["tau_mem"] == pytest.approx(200.0 / 20.0)
    assert iaf["v_threshold"] == pytest.approx(10.0)
    assert iaf["v_reset"] == pytest.approx(-5.0)
    assert cells["iafref"].params["refractory_period"] == 4
    assert cells["tau"].params["tau_mem"] == pytest.approx(20.0)
    assert cells["tauref"].params["refractory_period"] == 3
    assert (cells["izh"].params["a"], cells["izh"].params["d"]) == (0.02, 8.0)
    izh07 = cells["izh07"].params
    assert izh07["C"] == pytest.approx(100.0)
    assert izh07["k"] == pytest.approx(0.7)
    assert izh07["a"] == pytest.approx(0.03)
    assert izh07["b"] == pytest.approx(-2.0)
    assert izh07["d"] == pytest.approx(100.0)
    adex = cells["adex"].params
    assert adex["tau"] == pytest.approx(281.0 / 30.0)
    assert adex["tau_w"] == pytest.approx(144.0)
    assert adex["b"] == pytest.approx(80.5)
    for cell in cells.values():
        assert cell.notes and cell.notes[0].startswith("NeuroML cells carry no timestep")
        create_neuron(cell)


def test_a_libneuroml_network_document_is_refused_not_trimmed_to_its_cells(
    tmp_path: Path,
) -> None:
    """A document describing a network is not reduced to a list of cells."""
    document = neuroml.NeuroMLDocument(id="net_doc")
    document.add(
        neuroml.IafTauCell,
        id="cell",
        tau="20ms",
        leak_reversal="-65mV",
        thresh="-55mV",
        reset="-70mV",
    )
    network = document.add(neuroml.Network, id="net", validate=False)
    network.add(neuroml.Population, id="pop", component="cell", size=3)

    with pytest.raises(ValueError, match="does not model: network"):
        import_neuroml(_write(document, tmp_path / "net.nml"))
