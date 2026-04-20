# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for core/types

module TypesAccel

using Statistics, LinearAlgebra

mutable struct LayerSpecState
    max_luts::Float64
    max_ffs::Float64
    max_bram_kb::Float64
    max_dsp::Float64
    max_power_mw::Float64
    max_latency_cycles::Float64
    total_luts::Float64
    total_ffs::Float64
    total_dsp::Float64
    total_bram_kb::Float64
    total_power_mw::Float64
    total_latency_cycles::Float64
    mean_accuracy::Float64
    layer_id::Float64
    neurons::Float64
end

function LayerSpecState()
    LayerSpecState(500000.0, 500000.0, 2048.0, 256.0, 5000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 64.0)
end

function utilisation(s::LayerSpecState)
    bram: int = 0, dsp: int = 0) -> Dict[str, float]
    return {
    "luts": luts / s.max_luts if s.max_luts else 0,
    "ffs": ffs / s.max_ffs if s.max_ffs else 0,
    "bram": bram / s.max_bram_kb if s.max_bram_kb else 0,
    "dsp": dsp / s.max_dsp if s.max_dsp else 0,
    }
end

function meets_budget(s::LayerSpecState, budget)
    if s.total_luts > budget.max_luts
        return false
    if s.total_power_mw > budget.max_power_mw
        return false
    if budget.max_latency_cycles > 0 && s.total_latency_cycles > budget.max_latency_cycles
        return false
    if s.total_ffs > budget.max_ffs
        return false
    if s.total_dsp > budget.max_dsp
        return false
    if s.total_bram_kb > budget.max_bram_kb
        return false
    return true
end

function summary(s::LayerSpecState)
    return (
        f"LUTs: {s.total_luts}, FFs: {s.total_ffs}, "
        f"DSP: {s.total_dsp}, BRAM: {s.total_bram_kb:.1f} KB, "
        f"Power: {s.total_power_mw:.2f} mW, "
        f"Latency: {s.total_latency_cycles} cycles, "
        f"Accuracy: {s.mean_accuracy:.4f}"
    )
end

function estimate_luts(s::LayerSpecState)
    if s.mode == ComputeMode.DETERMINISTIC
        return max(s.mac_count, s.neurons) * 120
    base_macs = max(s.mac_count, s.neurons * 2)
    luts = base_macs * 2 + int(math.log2(max(1, s.bitstream_length))) * 5
    decorr_cost = {
        DecorrelationStrategy.SOBOL: base_macs * 15,
        DecorrelationStrategy.HALTON: base_macs * 12,
        DecorrelationStrategy.SCC_DECORRELATOR: base_macs * 8,
        DecorrelationStrategy.LFSR: 16,
    }.get(s.decorrelator, 0)
    luts += decorr_cost
    neuron_mult = {
        NeuronType.LIF: 1.0,
        NeuronType.IZHIKEVICH: 1.8,
        NeuronType.ADEX: 2.2,
        NeuronType.HH: 4.5,
    }.get(s.neuron_type, 1.0)
    return int(luts * neuron_mult)
end

function estimate_power_mw(s::LayerSpecState)
    if s.mode == ComputeMode.DETERMINISTIC
        return max(s.mac_count, s.neurons) * 0.5
    base = max(s.mac_count, s.neurons)
    return base * 0.01 * (s.bitstream_length / 256.0)
end

function estimate_accuracy(s::LayerSpecState)
    if s.mode == ComputeMode.DETERMINISTIC
        return 1.0
    length = max(1, s.bitstream_length)
    base = {
        DecorrelationStrategy.SOBOL: 1.0 - 1.0 / length,
        DecorrelationStrategy.HALTON: 1.0 - 1.2 / length,
        DecorrelationStrategy.SCC_DECORRELATOR: 1.0 - 1.5 / length,
        DecorrelationStrategy.LFSR: 1.0 - 1.0 / math.sqrt(length),
    }.get(s.decorrelator, 1.0 - 2.0 / math.sqrt(length))
    return max(0.1, min(1.0, base))
end

function estimate_network(layers)
    return ResourceReport(
        total_luts=sum(l.estimate_luts() for l in layers),
        total_power_mw=sum(l.estimate_power_mw() for l in layers),
        total_latency_cycles=max((l.bitstream_length for l in layers), default=0),
        mean_accuracy=sum(l.estimate_accuracy() for l in layers) / max(length(layers), 1),
    )
end

end # module TypesAccel
