# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Folded-interconnect area estimator tests

"""The folded area estimator maps FoldedResourceMetrics onto the calibrated
per-block FPGA costs, reproducing each modelled term exactly and inheriting the
target capacity / power / latency conventions of the energy estimator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sc_neurocore.energy.folded_estimator import FoldedAreaEstimate, estimate_folded_area
from sc_neurocore.energy.fpga_models import AXI_LITE, EVENT_NEURON, LIF_NEURON, TARGETS
from sc_neurocore.nir_bridge.fpga_compiler import FoldedResourceMetrics

_DW = 16


def _metrics(
    *,
    neurons: int = 64,
    pe_instances: int = 1,
    shared_multipliers: int = 8,
    populations: int = 2,
    state_vars: int = 1,
) -> FoldedResourceMetrics:
    return FoldedResourceMetrics(
        neurons=neurons,
        state_vars_per_neuron=state_vars,
        pe_instances=pe_instances,
        shared_multipliers=shared_multipliers,
        state_ram_bits=neurons * state_vars * _DW,
        cycles_per_tick=neurons + 1,
        direct_neuron_instances=neurons,
        populations=populations,
    )


def test_pe_pool_luts_match_calibrated_neuron_cost() -> None:
    est = estimate_folded_area(_metrics(pe_instances=3), target="ice40", data_width=_DW)
    assert est.pe_luts == 3 * LIF_NEURON.luts  # one combinational PE per distinct type


def test_event_driven_uses_event_neuron_and_aer_infra() -> None:
    m = _metrics(pe_instances=2)
    lif = estimate_folded_area(m, target="ice40", data_width=_DW)
    evt = estimate_folded_area(m, target="ice40", data_width=_DW, event_driven=True)
    assert evt.pe_luts == 2 * EVENT_NEURON.luts
    assert evt.pe_luts < lif.pe_luts  # the event neuron is cheaper
    assert evt.infra_luts > lif.infra_luts  # AER encoder + router added
    # Lower switching activity ⇒ lower dynamic power for the same total LUTs scale.
    assert evt.dynamic_power_mw < lif.dynamic_power_mw


def test_dsp_target_uses_one_dsp_per_multiplier_no_lut_multiply() -> None:
    m = _metrics(shared_multipliers=8)
    art = estimate_folded_area(m, target="artix7", data_width=_DW)
    assert art.dsps == 8  # artix7 has DSP slices
    assert art.multiplier_luts == 0


def test_dspless_target_uses_lut_multiply_no_dsp() -> None:
    m = _metrics(shared_multipliers=8)
    ice = estimate_folded_area(m, target="ice40", data_width=_DW)
    assert ice.dsps == 0  # ice40 has no DSP slices
    assert ice.multiplier_luts == 8 * ((_DW * _DW) // 4)


def test_rom_luts_scale_with_multipliers_and_log_neurons() -> None:
    m = _metrics(neurons=64, shared_multipliers=5)
    est = estimate_folded_area(m, target="ice40", data_width=_DW)
    depth = max(1, math.ceil(math.log2(64)))
    assert est.rom_luts == 5 * _DW * depth


def test_no_multiplier_no_rom_for_pure_spiking_network() -> None:
    # A recurrent/inter-pop spiking network has zero shared multipliers (spike-gated),
    # so neither the multiplier nor the weight-ROM terms contribute.
    m = _metrics(shared_multipliers=0)
    est = estimate_folded_area(m, target="ice40", data_width=_DW)
    assert est.multiplier_luts == 0
    assert est.rom_luts == 0
    assert est.dsps == 0


def test_flip_flops_cover_spike_bus_double_buffer_and_sequencer() -> None:
    m = _metrics(neurons=64, populations=2)
    est = estimate_folded_area(m, target="ice40", data_width=_DW)
    assert est.spike_bus_ffs == 2 * 64  # spike accumulator + committed spike bus
    # pidx (log2 populations) + nidx (log2 neurons) + phase + tick_done
    assert est.sequencer_ffs == 1 + 6 + 2
    assert est.total_ffs == est.spike_bus_ffs + est.sequencer_ffs


def test_bram_kb_from_state_ram_bits() -> None:
    m = _metrics(neurons=100, state_vars=2)  # 100 * 2 * 16 = 3200 bits
    est = estimate_folded_area(m, target="ecp5", data_width=_DW)
    assert est.bram_bits == 3200
    assert est.total_bram_kb == pytest.approx(3200 / 8192.0)


def test_infra_excluded_when_requested() -> None:
    m = _metrics()
    with_infra = estimate_folded_area(m, target="ice40", data_width=_DW)
    without = estimate_folded_area(m, target="ice40", data_width=_DW, include_infra=False)
    assert with_infra.infra_luts == AXI_LITE.luts
    assert without.infra_luts == 0
    assert without.total_luts == with_infra.total_luts - AXI_LITE.luts


def test_latency_is_cycles_per_tick_and_drives_energy() -> None:
    m = _metrics(neurons=64)  # cycles_per_tick = 65
    est = estimate_folded_area(m, target="ice40", data_width=_DW, clock_mhz=100.0)
    assert est.latency_cycles == 65
    expected_nj = est.dynamic_power_mw * (65 / (100.0 * 1e6)) * 1e6
    assert est.energy_per_tick_nj == pytest.approx(expected_nj)


def test_fit_and_utilisation_against_target_budget() -> None:
    small = estimate_folded_area(_metrics(neurons=16, shared_multipliers=2), target="ice40")
    assert small.fits_on_target
    assert 0.0 < small.lut_utilisation_pct < 100.0
    # A network whose LUT-based multipliers blow past the ice40 budget does not fit.
    huge = estimate_folded_area(
        _metrics(neurons=4096, shared_multipliers=4096), target="ice40", data_width=_DW
    )
    assert not huge.fits_on_target
    assert huge.lut_utilisation_pct > 100.0


def test_dsp_overflow_marks_unfit_even_when_luts_fit() -> None:
    # More multipliers than the target's DSP slices ⇒ does not fit, despite small LUTs.
    m = _metrics(neurons=8, shared_multipliers=TARGETS["artix7"].total_dsp + 1)
    est = estimate_folded_area(m, target="artix7", data_width=_DW)
    assert est.dsps > TARGETS["artix7"].total_dsp
    assert not est.fits_on_target


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
