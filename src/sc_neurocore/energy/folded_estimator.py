# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Folded-interconnect FPGA resource estimator

"""Pre-synthesis area, latency, and power estimate for the folded interconnect.

The folded (time-multiplexed) interconnect shares one combinational processing
element per neuron *type* across every neuron of that type, holds per-neuron state
in BRAM, and walks the network one neuron per cycle. Its resource profile is
therefore unlike the fully-unrolled direct path that :mod:`sc_neurocore.energy.estimator`
models (one neuron datapath per neuron); this module maps the architectural counts
in a :class:`~sc_neurocore.nir_bridge.fpga_compiler.FoldedResourceMetrics` summary
onto the same Yosys-calibrated per-block costs in :mod:`sc_neurocore.energy.fpga_models`,
so the folded estimate inherits those primitives' accuracy (~20% of synthesis) rather
than introducing new, uncalibrated coefficients.

Modelled resource terms (each from a calibrated primitive or the existing mux/multiply
convention reused from :func:`sc_neurocore.energy.estimator.estimate`):

* **PE pool** — one combinational neuron datapath per distinct type
  (``pe_instances`` × the per-neuron LUT cost). The per-neuron state flip-flops of the
  unrolled neuron are *not* charged: folded state lives in BRAM, not registers.
* **Weighted-fan-in multipliers** — ``shared_multipliers`` signed ``data_width`` ×
  ``data_width`` products (external and analogue-voltage source columns; spiking fan-in
  is spike-gated and uses none). One DSP each on a target with DSP slices, else a
  LUT-based multiply (``data_width² / 4`` LUTs).
* **Weight/threshold/bias case-ROM** — the per-neuron constant ROM selected by the
  sequencer index: a ``data_width``-wide mux over ``neurons`` rows per multiplier column
  (``data_width × ⌈log₂ neurons⌉`` LUTs per column, the popcount-mux convention).
* **Spike-bus double-buffer** — the per-tick spike accumulator plus the committed spike
  bus, one flip-flop per neuron each (``2 × neurons``).
* **Sequencer** — the ``(population, neuron)`` index counters plus the phase/tick flags.
* **State BRAM** — ``state_ram_bits`` of packed per-neuron state.
* **Latency** — ``cycles_per_tick`` cycles to advance the whole network one timestep
  (one neuron per cycle plus the commit cycle), which the clock turns into seconds and,
  with the dynamic power, into energy per tick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .fpga_models import (
    TARGETS,
    LIF_NEURON,
    EVENT_NEURON,
    AXI_LITE,
    AER_ENCODER,
    AER_ROUTER,
)

if TYPE_CHECKING:
    from sc_neurocore.nir_bridge.fpga_compiler import FoldedResourceMetrics


@dataclass
class FoldedAreaEstimate:
    """Area, latency, and power estimate for a folded-interconnect network.

    All counts are pre-synthesis estimates derived from Yosys-calibrated per-block
    costs; treat them as ~20%-accurate architectural guidance, not synthesis truth.
    """

    target: str
    pe_luts: int
    multiplier_luts: int
    rom_luts: int
    infra_luts: int
    dsps: int
    bram_bits: int
    latency_cycles: int
    clock_freq_mhz: float
    dynamic_power_mw: float
    # Flip-flop terms (the per-neuron spike-bus double-buffer and the sequencer counters).
    spike_bus_ffs: int = 0
    sequencer_ffs: int = 0
    # Computed in __post_init__ from the fields above.
    total_luts: int = field(init=False)
    total_ffs: int = field(init=False)
    total_bram_kb: float = field(init=False)
    energy_per_tick_nj: float = field(init=False)
    fits_on_target: bool = field(init=False)
    lut_utilisation_pct: float = field(init=False)

    def __post_init__(self) -> None:
        self.total_luts = self.pe_luts + self.multiplier_luts + self.rom_luts + self.infra_luts
        self.total_ffs = self.spike_bus_ffs + self.sequencer_ffs
        self.total_bram_kb = self.bram_bits / 8192.0

        latency_s = self.latency_cycles / (self.clock_freq_mhz * 1e6)
        self.energy_per_tick_nj = self.dynamic_power_mw * latency_s * 1e6

        target_info = TARGETS.get(self.target)
        if target_info is not None:
            self.fits_on_target = (
                self.total_luts <= target_info.total_luts
                and self.dsps <= target_info.total_dsp
                and self.total_bram_kb <= target_info.total_bram_kb
            )
            self.lut_utilisation_pct = (self.total_luts / target_info.total_luts) * 100.0
        else:  # pragma: no cover — guarded by the caller (estimate_folded_area validates)
            self.fits_on_target = True
            self.lut_utilisation_pct = 0.0

    def as_dict(self) -> dict[str, float | int | str | bool]:
        """Return a plain mapping of the estimate (for JSON artefacts)."""
        return {
            "target": self.target,
            "total_luts": self.total_luts,
            "pe_luts": self.pe_luts,
            "multiplier_luts": self.multiplier_luts,
            "rom_luts": self.rom_luts,
            "infra_luts": self.infra_luts,
            "total_ffs": self.total_ffs,
            "dsps": self.dsps,
            "total_bram_kb": self.total_bram_kb,
            "latency_cycles": self.latency_cycles,
            "clock_freq_mhz": self.clock_freq_mhz,
            "dynamic_power_mw": self.dynamic_power_mw,
            "energy_per_tick_nj": self.energy_per_tick_nj,
            "fits_on_target": self.fits_on_target,
            "lut_utilisation_pct": self.lut_utilisation_pct,
        }

    def summary(self) -> str:
        """Human-readable one-block summary."""
        return "\n".join(
            [
                f"Folded area estimate — {self.target}",
                f"  PE pool:        {self.pe_luts} LUTs",
                f"  Multipliers:    {self.multiplier_luts} LUTs, {self.dsps} DSP",
                f"  Weight ROM:     {self.rom_luts} LUTs",
                f"  Infrastructure: {self.infra_luts} LUTs",
                f"  Total LUTs:     {self.total_luts:,}",
                f"  Total FFs:      {self.total_ffs:,}",
                f"  State BRAM:     {self.total_bram_kb:.2f} KB",
                f"  Latency:        {self.latency_cycles:,} cycles/tick",
                f"  Dynamic power:  {self.dynamic_power_mw:.2f} mW",
                f"  Energy/tick:    {self.energy_per_tick_nj:.3f} nJ",
                f"  LUT util:       {self.lut_utilisation_pct:.1f}%",
                f"  Fits on target: {'YES' if self.fits_on_target else 'NO — exceeds budget'}",
            ]
        )


def estimate_folded_area(
    metrics: FoldedResourceMetrics,
    *,
    target: str = "ice40",
    data_width: int = 16,
    clock_mhz: float = 100.0,
    event_driven: bool = False,
    include_infra: bool = True,
) -> FoldedAreaEstimate:
    """Estimate folded-interconnect FPGA resources, latency, and power.

    Parameters
    ----------
    metrics : FoldedResourceMetrics
        The architectural summary attached to a folded compile
        (``NetworkCompilationResult.folded_metrics``).
    target : str
        FPGA target key in :data:`sc_neurocore.energy.fpga_models.TARGETS`
        (``'ice40'``, ``'ecp5'``, ``'artix7'``, ``'zynq'``).
    data_width : int
        Fixed-point data width (the multiply and ROM-mux widths).
    clock_mhz : float
        Target clock frequency, used to turn cycles into time and energy.
    event_driven : bool
        Charge the event-driven neuron cost and AER infrastructure instead of the
        clock-driven LIF cost.
    include_infra : bool
        Add the AXI-Lite register-file (and AER, when ``event_driven``) infrastructure
        LUTs.

    Returns
    -------
    FoldedAreaEstimate
        The folded area, latency, and power estimate.

    Raises
    ------
    ValueError
        If ``target`` is not a known FPGA target.
    """
    target_info = TARGETS.get(target)
    if target_info is None:
        raise ValueError(f"Unknown target '{target}'. Options: {list(TARGETS)}")

    neuron_cost = EVENT_NEURON if event_driven else LIF_NEURON

    # One combinational neuron datapath per distinct type (the per-type PE pool); the
    # per-neuron state flip-flops are replaced by BRAM, so only the datapath LUTs count.
    pe_luts = metrics.pe_instances * neuron_cost.luts

    # Each shared weighted-fan-in multiply is one DSP on a DSP-bearing target, otherwise
    # a LUT-based data_width × data_width multiplier.
    if target_info.total_dsp > 0:
        dsps = metrics.shared_multipliers
        multiplier_luts = 0
    else:
        dsps = 0
        multiplier_luts = metrics.shared_multipliers * ((data_width * data_width) // 4)

    # The per-neuron weight/threshold/bias ROM is a data_width-wide mux over `neurons`
    # rows for each multiplier column (the popcount-mux convention from estimate()).
    rom_mux_depth = max(1, math.ceil(math.log2(max(metrics.neurons, 2))))
    rom_luts = metrics.shared_multipliers * data_width * rom_mux_depth

    # Spike accumulator + committed spike bus: one flip-flop per neuron each.
    spike_bus_ffs = 2 * metrics.neurons
    # (population, neuron) index counters plus the phase and tick_done flags.
    pop_idx_w = max(1, (max(metrics.populations - 1, 1)).bit_length())
    neuron_idx_w = max(1, (max(metrics.neurons - 1, 1)).bit_length())
    sequencer_ffs = pop_idx_w + neuron_idx_w + 2

    infra_luts = 0
    if include_infra:
        infra_luts = AXI_LITE.luts
        if event_driven:
            infra_luts += AER_ENCODER.luts + AER_ROUTER.luts

    total_luts = pe_luts + multiplier_luts + rom_luts + infra_luts

    # Dynamic power: C_eff × V² × f × N_LUTs × activity (same model as estimate()).
    activity = 0.1 if event_driven else 0.5
    c_eff_f = target_info.c_eff_per_lut_ff * 1e-15
    v_sq = target_info.voltage**2
    freq = clock_mhz * 1e6
    dynamic_power_mw = c_eff_f * v_sq * freq * total_luts * activity * 1e3

    return FoldedAreaEstimate(
        target=target,
        pe_luts=pe_luts,
        multiplier_luts=multiplier_luts,
        rom_luts=rom_luts,
        infra_luts=infra_luts,
        dsps=dsps,
        bram_bits=metrics.state_ram_bits,
        latency_cycles=metrics.cycles_per_tick,
        clock_freq_mhz=clock_mhz,
        dynamic_power_mw=dynamic_power_mw,
        spike_bus_ffs=spike_bus_ffs,
        sequencer_ffs=sequencer_ffs,
    )
