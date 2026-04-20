# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for energy/estimator

module EstimatorAccel

using Statistics, LinearAlgebra

mutable struct EnergyReportState
    name::Float64
    n_inputs::Float64
    n_neurons::Float64
    n_synapses::Float64
    bitstream_length::Float64
    luts::Float64
    ffs::Float64
    bram_bits::Float64
    dynamic_power_mw::Float64
    latency_cycles::Float64
    target::Float64
    layers::Float64
    total_luts::Float64
    total_ffs::Float64
    total_bram_kb::Float64
end

function EnergyReportState()
    EnergyReportState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function summary(s::EnergyReportState)
    lines = [
        f"SC-NeuroCore Energy Estimate — {s.target}",
        f"{'=' * 55}",
        "",
    ]
    for layer in s.layers
        lines = push!(,
            f"  {layer.name}: {layer.n_inputs}->{layer.n_neurons} "
            f"({layer.n_synapses} syn, L={layer.bitstream_length}) "
            f"-> {layer.luts} LUTs, {layer.dynamic_power_mw:.2f} mW"
        )
    lines.extend(
        [
            "",
            f"  Infrastructure: {s.infra_luts} LUTs",
            "",
            f"  Total LUTs:     {s.total_luts:,}",
            f"  Total FFs:      {s.total_ffs:,}",
            f"  Total BRAM:     {s.total_bram_kb:.1f} KB",
            f"  Dynamic power:  {s.total_dynamic_power_mw:.2f} mW",
            f"  Latency:        {s.total_latency_cycles:,} cycles",
            f"  Energy/inf:     {s.energy_per_inference_nj:.2f} nJ",
            f"  Clock:          {s.clock_freq_mhz:.0f} MHz",
            f"  Utilization:    {s.utilization_pct:.1f}%",
            f"  Fits on target: {'YES' if s.fits_on_target else 'NO — exceeds LUT budget'}",
        ]
    )
    return "\n".join(lines)
end

function estimate(layer_sizes, target, bitstream_length, neuron_type, event_driven, clock_mhz, include_infra)
    layer_sizes: list[tuple[int, int]],
    target: str = "ice40",
    bitstream_length: int = 256,
    neuron_type: str = "lif",
    event_driven: bool = false,
    clock_mhz: float = 100.0,
    include_infra: bool = true,
    ) -> EnergyReport
    target_info = TARGETS.get(target)
    if target_info is nothing
        raise ValueError(f"Unknown target '{target}'. Options: {list(TARGETS)}")
    neuron_cost = EVENT_NEURON if event_driven else LIF_NEURON
    layers = []
    for i, (n_in, n_out) in enumerate(layer_sizes)
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
        if n_synapses > 1024
            bram_bits = n_synapses * BRAM_BITS_PER_WEIGHT
        # Latency: L cycles for SC computation + 2 cycles for neuron update
        latency = bitstream_length + 2
        # Dynamic power: C_eff × V² × f × N_gates × activity
        # SC activity ~0.5 (random bitstreams toggle 50%)
        activity = 0.1 if event_driven else 0.5
        c_eff_f = target_info.c_eff_per_lut_ff * 1e-15
        v_sq = target_info.voltage^2
        freq = clock_mhz * 1e6
        power_w = c_eff_f * v_sq * freq * total_luts * activity
        power_mw = power_w * 1e3
        layers = push!(,
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
    if include_infra
        infra_luts = AXI_LITE.luts
        if event_driven
            infra_luts += AER_ENCODER.luts + AER_ROUTER.luts
    return EnergyReport(
        target=target,
        layers=layers,
        infra_luts=infra_luts,
        clock_freq_mhz=clock_mhz,
    )
end

end # module EstimatorAccel
