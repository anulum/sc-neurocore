# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (core_raster_parity) from former test_folded_interconnect.py

from __future__ import annotations

from tests.folded_interconnect_support import *  # noqa: F403


def test_folded_matches_direct_spike_raster() -> None:
    ng = _single_lif_graph()
    q = Q88(data_width=_DW, fraction=_FR)
    qgraph = quantise_graph(ng, q)
    flat = _flat_current_literal()

    direct_top = _build_top_direct("sc_fold_test", qgraph, data_width=_DW, fraction=_FR)
    lif_module = _build_neuron_module("lif", qgraph.populations[0], data_width=_DW, fraction=_FR)
    pe_modules, folded_top = _build_top_folded(
        "sc_fold_test_folded", qgraph, data_width=_DW, fraction=_FR
    )
    pe_source = "\n\n".join(pe_modules.values())

    direct_raster = _cosim({"lif": lif_module, "top": direct_top, "tb": _direct_tb(flat)}, "direct")
    folded_raster = _cosim({"pe": pe_source, "top": folded_top, "tb": _folded_tb(flat)}, "folded")

    assert len(direct_raster) == _STEPS, f"direct produced {len(direct_raster)} rows"
    assert len(folded_raster) == _STEPS, f"folded produced {len(folded_raster)} rows"
    assert folded_raster == direct_raster, (
        "folded spike raster diverged from direct:\n"
        f"  first mismatch at step "
        f"{next((i for i, (a, b) in enumerate(zip(direct_raster, folded_raster)) if a != b), None)}"
    )
    # Non-vacuous: the workload spikes.
    assert any("1" in row for row in direct_raster), "test workload should produce spikes"


def test_folded_weighted_external_matches_direct() -> None:
    ng, currents, n_dst = _weighted_ff_graph()
    q = Q88(data_width=_DW, fraction=_FR)
    qgraph = quantise_graph(ng, q)

    mask = (1 << _DW) - 1
    packed = 0
    for k, cur in enumerate(currents):
        packed |= (q.encode(cur) & mask) << (k * _DW)
    flat = f"{len(currents) * _DW}'h{packed:x}"

    direct_top = _build_top_direct("sc_fold_test", qgraph, data_width=_DW, fraction=_FR)
    lif_module = _build_neuron_module("lif", qgraph.populations[0], data_width=_DW, fraction=_FR)
    pe_modules, folded_top = _build_top_folded(
        "sc_fold_test_folded", qgraph, data_width=_DW, fraction=_FR
    )
    pe_source = "\n\n".join(pe_modules.values())

    direct_raster = _cosim(
        {"lif": lif_module, "top": direct_top, "tb": _direct_tb(flat, n_dst)}, "direct"
    )
    folded_raster = _cosim(
        {"pe": pe_source, "top": folded_top, "tb": _folded_tb(flat, n_dst)}, "folded"
    )

    assert len(direct_raster) == _STEPS and len(folded_raster) == _STEPS
    assert folded_raster == direct_raster, (
        "folded weighted-input raster diverged from direct at step "
        f"{next((i for i, (a, b) in enumerate(zip(direct_raster, folded_raster)) if a != b), None)}"
    )
    assert any("1" in row for row in direct_raster), "weighted workload should spike"


def test_folded_recurrent_matches_direct() -> None:
    ng, currents, n_dst = _recurrent_graph()
    q = Q88(data_width=_DW, fraction=_FR)
    qgraph = quantise_graph(ng, q)

    mask = (1 << _DW) - 1
    packed = 0
    for k, cur in enumerate(currents):
        packed |= (q.encode(cur) & mask) << (k * _DW)
    flat = f"{len(currents) * _DW}'h{packed:x}"

    direct_top = _build_top_direct("sc_fold_test", qgraph, data_width=_DW, fraction=_FR)
    lif_module = _build_neuron_module("lif", qgraph.populations[0], data_width=_DW, fraction=_FR)
    pe_modules, folded_top = _build_top_folded(
        "sc_fold_test_folded", qgraph, data_width=_DW, fraction=_FR
    )
    pe_source = "\n\n".join(pe_modules.values())

    direct_raster = _cosim(
        {"lif": lif_module, "top": direct_top, "tb": _direct_tb(flat, n_dst)}, "direct"
    )
    folded_raster = _cosim(
        {"pe": pe_source, "top": folded_top, "tb": _folded_tb(flat, n_dst)}, "folded"
    )

    assert len(direct_raster) == _STEPS and len(folded_raster) == _STEPS
    assert folded_raster == direct_raster, (
        "folded recurrent raster diverged from direct at step "
        f"{next((i for i, (a, b) in enumerate(zip(direct_raster, folded_raster)) if a != b), None)}"
    )
    assert any("1" in row for row in direct_raster), "recurrent workload should spike"


def test_folded_state_is_bram_backed_and_shares_one_pe() -> None:
    ng = _single_lif_graph()
    q = Q88(data_width=_DW, fraction=_FR)
    qgraph = quantise_graph(ng, q)
    pe_modules, folded_top = _build_top_folded(
        "sc_fold_test_folded", qgraph, data_width=_DW, fraction=_FR
    )
    # Shared datapath: exactly one PE instance, BRAM state, a sequencer — no per-neuron unroll.
    assert folded_top.count("pe_inst") == 1
    assert 'ram_style = "block"' in folded_top
    assert "state_bram" in folded_top
    assert "p0_n0_inst" not in folded_top  # the direct per-neuron instance name must be absent
    assert set(pe_modules) == {"lif_pe"}  # one PE module per distinct neuron type
