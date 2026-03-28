# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Pre-silicon energy estimator

"""Estimate FPGA resource usage, power, and energy before synthesis.

Takes an SNN architecture description (layer sizes, neuron types,
connectivity) and an FPGA target, returns estimated LUTs, BRAM,
dynamic power (mW), and energy per inference (nJ) in <1 second.

Calibrated against Yosys synth_ice40 reports for SC-NeuroCore HDL.
Accuracy target: within 20% of actual synthesis for our modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fpga_models import (
    TARGETS,
    LIF_NEURON,
    EVENT_NEURON,
    BITSTREAM_ENCODER,
    SC_SYNAPSE,
    AXI_LITE,
    AER_ENCODER,
    AER_ROUTER,
    BRAM_BITS_PER_WEIGHT,
)


@dataclass
class LayerEstimate:
    """Resource estimate for one layer."""

    name: str
    n_inputs: int
    n_neurons: int
    n_synapses: int
    bitstream_length: int
    luts: int
    ffs: int
    bram_bits: int
    dynamic_power_mw: float
    latency_cycles: int


@dataclass
class EnergyReport:
    """Complete pre-silicon energy estimate for an SNN on an FPGA target."""

    target: str
    layers: list[LayerEstimate]
    total_luts: int = field(init=False)
    total_ffs: int = field(init=False)
    total_bram_kb: float = field(init=False)
    infra_luts: int = 0
    total_dynamic_power_mw: float = field(init=False)
    total_latency_cycles: int = field(init=False)
    energy_per_inference_nj: float = field(init=False)
    clock_freq_mhz: float = 100.0
    fits_on_target: bool = field(init=False)
    utilization_pct: float = field(init=False)

    def __post_init__(self) -> None:
        self.total_luts = sum(l.luts for l in self.layers) + self.infra_luts
        self.total_ffs = sum(l.ffs for l in self.layers)
        self.total_bram_kb = sum(l.bram_bits for l in self.layers) / 8192.0
        self.total_dynamic_power_mw = sum(l.dynamic_power_mw for l in self.layers)
        self.total_latency_cycles = sum(l.latency_cycles for l in self.layers)

        latency_s = self.total_latency_cycles / (self.clock_freq_mhz * 1e6)
        self.energy_per_inference_nj = self.total_dynamic_power_mw * latency_s * 1e6

        target_info = TARGETS.get(self.target)
        if target_info:
            self.fits_on_target = self.total_luts <= target_info.total_luts
            self.utilization_pct = (self.total_luts / target_info.total_luts) * 100
        else:  # pragma: no cover — unknown target fallback
            self.fits_on_target = True
            self.utilization_pct = 0.0

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"SC-NeuroCore Energy Estimate — {self.target}",
            f"{'=' * 55}",
            "",
        ]
        for layer in self.layers:
            lines.append(
                f"  {layer.name}: {layer.n_inputs}->{layer.n_neurons} "
                f"({layer.n_synapses} syn, L={layer.bitstream_length}) "
                f"-> {layer.luts} LUTs, {layer.dynamic_power_mw:.2f} mW"
            )
        lines.extend(
            [
                "",
                f"  Infrastructure: {self.infra_luts} LUTs",
                "",
                f"  Total LUTs:     {self.total_luts:,}",
                f"  Total FFs:      {self.total_ffs:,}",
                f"  Total BRAM:     {self.total_bram_kb:.1f} KB",
                f"  Dynamic power:  {self.total_dynamic_power_mw:.2f} mW",
                f"  Latency:        {self.total_latency_cycles:,} cycles",
                f"  Energy/inf:     {self.energy_per_inference_nj:.2f} nJ",
                f"  Clock:          {self.clock_freq_mhz:.0f} MHz",
                f"  Utilization:    {self.utilization_pct:.1f}%",
                f"  Fits on target: {'YES' if self.fits_on_target else 'NO — exceeds LUT budget'}",
            ]
        )
        return "\n".join(lines)


def estimate(
    layer_sizes: list[tuple[int, int]],
    target: str = "ice40",
    bitstream_length: int = 256,
    neuron_type: str = "lif",
    event_driven: bool = False,
    clock_mhz: float = 100.0,
    include_infra: bool = True,
) -> EnergyReport:
    """Estimate FPGA resources and power for an SNN.

    Parameters
    ----------
    layer_sizes : list of (n_inputs, n_neurons)
        Architecture as list of layer dimensions.
    target : str
        FPGA target: 'ice40', 'ecp5', 'artix7', 'zynq'.
    bitstream_length : int
        SC bitstream length L (affects latency and precision).
    neuron_type : str
        'lif' (clock-driven) or 'event' (event-driven).
    event_driven : bool
        Use event-driven architecture (AER).
    clock_mhz : float
        Target clock frequency.
    include_infra : bool
        Include AXI/DMA infrastructure cost.

    Returns
    -------
    EnergyReport
        Complete resource and power estimate.
    """
    target_info = TARGETS.get(target)
    if target_info is None:
        raise ValueError(f"Unknown target '{target}'. Options: {list(TARGETS)}")

    neuron_cost = EVENT_NEURON if event_driven else LIF_NEURON

    layers = []
    for i, (n_in, n_out) in enumerate(layer_sizes):
        n_synapses = n_in * n_out
        n_encoders = n_in

        # LUT cost
        luts_neurons = n_out * neuron_cost.luts
        luts_synapses = n_synapses * SC_SYNAPSE.luts
        luts_encoders = n_encoders * BITSTREAM_ENCODER.luts
        # MUX trees for popcount: ~log2(n_in) LUTs per neuron
        luts_mux = n_out * max(1, int(np.log2(max(n_in, 2))))
        total_luts = luts_neurons + luts_synapses + luts_encoders + luts_mux

        # FF cost
        ffs = n_out * neuron_cost.ffs + n_encoders * BITSTREAM_ENCODER.ffs

        # BRAM for weights (if too many for LUT registers)
        bram_bits = 0
        if n_synapses > 1024:
            bram_bits = n_synapses * BRAM_BITS_PER_WEIGHT

        # Latency: L cycles for SC computation + 2 cycles for neuron update
        latency = bitstream_length + 2

        # Dynamic power: C_eff × V² × f × N_gates × activity
        # SC activity ~0.5 (random bitstreams toggle 50%)
        activity = 0.1 if event_driven else 0.5
        c_eff_f = target_info.c_eff_per_lut_ff * 1e-15
        v_sq = target_info.voltage**2
        freq = clock_mhz * 1e6
        power_w = c_eff_f * v_sq * freq * total_luts * activity
        power_mw = power_w * 1e3

        layers.append(
            LayerEstimate(
                name=f"layer_{i}",
                n_inputs=n_in,
                n_neurons=n_out,
                n_synapses=n_synapses,
                bitstream_length=bitstream_length,
                luts=total_luts,
                ffs=ffs,
                bram_bits=bram_bits,
                dynamic_power_mw=power_mw,
                latency_cycles=latency,
            )
        )

    # Infrastructure cost
    infra_luts = 0
    if include_infra:
        infra_luts = AXI_LITE.luts
        if event_driven:
            infra_luts += AER_ENCODER.luts + AER_ROUTER.luts

    return EnergyReport(
        target=target,
        layers=layers,
        infra_luts=infra_luts,
        clock_freq_mhz=clock_mhz,
    )
