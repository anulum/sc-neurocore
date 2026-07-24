# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (multi_population_compile) from former test_folded_interconnect.py

from __future__ import annotations

from tests.folded_interconnect_support import *  # noqa: F403


def test_folded_two_population_feedforward_matches_direct() -> None:
    ng, currents, n_total = _two_pop_ff_graph()
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
        {"lif": lif_module, "top": direct_top, "tb": _direct_tb(flat, n_total)}, "direct"
    )
    folded_raster = _cosim(
        {"pe": pe_source, "top": folded_top, "tb": _folded_tb(flat, n_total)}, "folded"
    )

    assert len(direct_raster) == _STEPS and len(folded_raster) == _STEPS
    assert folded_raster == direct_raster, (
        "folded two-population feedforward raster diverged from direct at step "
        f"{next((i for i, (a, b) in enumerate(zip(direct_raster, folded_raster)) if a != b), None)}"
    )
    # Both layers must be exercised. spike_bus is [6:0] printed MSB-first: the output
    # pop occupies bits [4:6] (the leading 3 chars), the input pop bits [0:3] (trailing 4).
    assert any("1" in row[3:] for row in direct_raster), "input population should spike"
    assert any("1" in row[:3] for row in direct_raster), "output population should spike"


def test_folded_two_population_recurrent_matches_direct() -> None:
    ng, currents, n_total = _two_pop_recurrent_graph()
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
        {"lif": lif_module, "top": direct_top, "tb": _direct_tb(flat, n_total)}, "direct"
    )
    folded_raster = _cosim(
        {"pe": pe_source, "top": folded_top, "tb": _folded_tb(flat, n_total)}, "folded"
    )

    assert len(direct_raster) == _STEPS and len(folded_raster) == _STEPS
    assert folded_raster == direct_raster, (
        "folded two-population recurrent raster diverged from direct at step "
        f"{next((i for i, (a, b) in enumerate(zip(direct_raster, folded_raster)) if a != b), None)}"
    )
    assert any("1" in row for row in direct_raster), "recurrent workload should spike"


def test_folded_multi_population_shares_one_pe_per_type() -> None:
    ng, _currents, _n = _two_pop_ff_graph()
    q = Q88(data_width=_DW, fraction=_FR)
    qgraph = quantise_graph(ng, q)
    pe_modules, folded_top = _build_top_folded(
        "sc_fold_test_folded", qgraph, data_width=_DW, fraction=_FR
    )
    # Two LIF populations collapse to one shared PE instance and a per-population BRAM.
    assert set(pe_modules) == {"lif_pe"}
    assert folded_top.count("pe_inst_lif") == 1
    assert "state_bram_0" in folded_top and "state_bram_1" in folded_top
    assert "p0_n0_inst" not in folded_top  # no direct per-neuron unrolling


def test_compile_network_folded_two_populations() -> None:
    from sc_neurocore.nir_bridge.fpga_compiler import compile_network_to_fpga

    ng, _currents, n_total = _two_pop_ff_graph()
    result = compile_network_to_fpga(ng, interconnect="folded")
    assert result.interconnect == "folded"
    assert "lif_pe" in result.neuron_modules
    assert result.folded_metrics is not None
    assert result.folded_metrics.populations == 2
    assert result.folded_metrics.neurons == n_total
    assert result.folded_metrics.pe_instances == 1  # one PE shared across both LIF pops


def test_compile_network_folded_opt_in() -> None:
    from sc_neurocore.nir_bridge.fpga_compiler import compile_network_to_fpga

    result = compile_network_to_fpga(_single_lif_graph(), interconnect="folded")
    assert result.interconnect == "folded"
    assert "lif_pe" in result.neuron_modules  # the shared datapath PE was emitted
    assert "state_bram" in result.top_module


def test_compile_network_folded_analogue_source() -> None:
    import numpy as np

    from sc_neurocore.nir_bridge.fpga_compiler import compile_network_to_fpga

    # An analogue li source feeding a lif population folds, emitting the global voltage
    # bus; the source's three columns are counted as shared multipliers (v * weight).
    li = NeuronSpec(name="a", neuron_type="li", n_neurons=3, params={}, dt=1.0)
    lif = NeuronSpec(name="b", neuron_type="lif", n_neurons=3, params={}, dt=1.0)
    ng = NeuronGraph(
        populations=[li, lif],
        connections=[ConnectionSpec(src="a", dst="b", weights=np.full((3, 3), 0.5, np.float32))],
        input_pop="a",
        output_pop="b",
        dt=1.0,
    )
    result = compile_network_to_fpga(ng, interconnect="folded")
    assert result.interconnect == "folded"
    assert "v_bus" in result.top_module  # the analogue voltage double-buffer was emitted
    assert {"li_pe", "lif_pe"} <= set(result.neuron_modules)
    assert result.folded_metrics is not None
    assert result.folded_metrics.populations == 2
    assert result.folded_metrics.pe_instances == 2  # distinct types: li + lif
    assert result.folded_metrics.shared_multipliers == 3  # the analogue source's 3 columns


def test_compile_network_folded_rejects_unsupported_graph() -> None:
    from sc_neurocore.nir_bridge.fpga_compiler import compile_network_to_fpga

    # A *delayed external* source connection is outside the folded subset: a synaptic
    # delay has registered semantics only from a neuron population (spike_bus_hist or
    # v_bus_hist), so a delayed non-population input falls back to direct. Delayed
    # spiking and delayed analogue population sources both fold.
    import numpy as np

    pop = NeuronSpec(name="b", neuron_type="lif", n_neurons=4, params={}, dt=1.0)
    delayed_external = ConnectionSpec(
        src="stim",
        dst="b",
        weights=np.ones((4, 3), np.float32) * 0.4,
        delay_steps=2,
    )
    ng = NeuronGraph(
        populations=[pop],
        connections=[delayed_external],
        input_pop="stim",
        output_pop="b",
        dt=1.0,
    )
    with pytest.raises(ValueError, match="folded"):
        compile_network_to_fpga(ng, interconnect="folded")


def test_folded_two_population_delayed_feedforward_matches_direct() -> None:
    ng, currents, n_total = _two_pop_delayed_ff_graph()
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
        {"lif": lif_module, "top": direct_top, "tb": _direct_tb(flat, n_total)}, "direct"
    )
    folded_raster = _cosim(
        {"pe": pe_source, "top": folded_top, "tb": _folded_tb(flat, n_total)}, "folded"
    )

    assert len(direct_raster) == _STEPS and len(folded_raster) == _STEPS
    assert folded_raster == direct_raster, (
        "folded delayed two-population raster diverged from direct at step "
        f"{next((i for i, (a, b) in enumerate(zip(direct_raster, folded_raster)) if a != b), None)}"
    )
    assert any("1" in row[:3] for row in direct_raster), "output population should spike"


def test_folded_two_population_source_threshold_matches_direct() -> None:
    import numpy as np

    # Inter-population spiking fan-in gated by a source Threshold: the spike magnitude
    # (1.0 in Q8.8) is compared against the per-column threshold before the weight gates.
    n_in, n_out = 4, 3
    ext = np.array([[1.4, 1.0], [1.6, 0.8], [1.2, 1.2], [1.8, 0.6]], dtype=np.float32)
    ff = np.array(
        [[2.0, 0.0, 1.8, 1.2], [1.2, 2.0, 0.0, 1.8], [1.6, 1.6, 1.4, 0.0]], dtype=np.float32
    )
    # Thresholds below the 1.0 spike magnitude so a spike passes the gate (column 1 above
    # it, so that column never contributes — exercising both sides of the comparator).
    src_thr = np.array([0.5, 1.5, 0.5, 0.5], dtype=np.float32)
    inp = NeuronSpec(name="inp", neuron_type="lif", n_neurons=n_in, params={}, dt=1.0)
    out = NeuronSpec(name="out", neuron_type="lif", n_neurons=n_out, params={}, dt=1.0)
    ng = NeuronGraph(
        populations=[inp, out],
        connections=[
            ConnectionSpec(src="stim", dst="inp", weights=ext),
            ConnectionSpec(src="inp", dst="out", weights=ff, source_threshold=src_thr),
        ],
        input_pop="stim",
        output_pop="out",
        dt=1.0,
    )
    direct_raster, folded_raster = _parity_rasters(ng, [4.0, 3.5], n_in + n_out)
    assert len(direct_raster) == _STEPS and len(folded_raster) == _STEPS
    assert folded_raster == direct_raster, (
        "folded inter-pop source-threshold raster diverged from direct at step "
        f"{next((i for i, (a, b) in enumerate(zip(direct_raster, folded_raster)) if a != b), None)}"
    )
    assert any("1" in row[:3] for row in direct_raster), "output population should spike"
