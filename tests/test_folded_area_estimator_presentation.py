# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Folded-area presentation and integration tests

"""Validation, summary, and real-compiler integration for folded area."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.energy.folded_estimator import FoldedAreaEstimate, estimate_folded_area
from sc_neurocore.energy.fpga_models import LIF_NEURON
from tests.folded_area_estimator_support import _DW, _metrics


def test_unknown_target_raises() -> None:
    with pytest.raises(ValueError, match="Unknown target"):
        estimate_folded_area(_metrics(), target="web")


def test_as_dict_and_summary_surface_the_estimate() -> None:
    est = estimate_folded_area(_metrics(), target="ice40", data_width=_DW)
    d = est.as_dict()
    assert d["target"] == "ice40"
    assert d["total_luts"] == est.total_luts
    assert d["latency_cycles"] == est.latency_cycles
    assert set(d) >= {"total_luts", "total_ffs", "dsps", "total_bram_kb", "fits_on_target"}
    text = est.summary()
    assert "LUTs" in text and "mW" in text and "ice40" in text


def test_estimate_from_real_folded_compile() -> None:
    # End-to-end: a real folded compile's metrics flow into the area estimate.
    from sc_neurocore.nir_bridge.fpga_compiler import compile_network_to_fpga
    from sc_neurocore.nir_bridge.neuron_graph import ConnectionSpec, NeuronGraph, NeuronSpec

    inp = NeuronSpec(name="inp", neuron_type="lif", n_neurons=4, params={}, dt=1.0)
    out = NeuronSpec(name="out", neuron_type="lif", n_neurons=3, params={}, dt=1.0)
    ng = NeuronGraph(
        populations=[inp, out],
        connections=[
            ConnectionSpec(src="stim", dst="inp", weights=np.full((4, 2), 0.5, np.float32)),
            ConnectionSpec(src="inp", dst="out", weights=np.full((3, 4), 0.5, np.float32)),
        ],
        input_pop="stim",
        output_pop="out",
        dt=1.0,
    )
    result = compile_network_to_fpga(ng, interconnect="folded")
    assert result.folded_metrics is not None
    est = estimate_folded_area(result.folded_metrics, target="ice40", data_width=16)
    assert isinstance(est, FoldedAreaEstimate)
    assert est.latency_cycles == result.folded_metrics.cycles_per_tick
    assert est.pe_luts == result.folded_metrics.pe_instances * LIF_NEURON.luts
    # Only the external 'stim'→'inp' connection (2 columns) multiplies; the inp→out
    # spiking fan-in is spike-gated, so the multiplier count matches the metrics.
    assert est.dsps == 0  # ice40
    assert est.total_luts > 0
