# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for debug/analyzer

module AnalyzerAccel

using Statistics, LinearAlgebra

mutable struct CausalEventState
    timestep::Float64
    neuron_id::Float64
    trace_a_spike::Float64
    trace_b_spike::Float64
    trace_a_voltage::Float64
    trace_b_voltage::Float64
    voltage_diff::Float64
    input_current::Float64
    voltage::Float64
    spiked::Float64
end

function CausalEventState()
    CausalEventState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function find_divergence(trace_a, trace_b)
    trace_a: ExecutionTrace,
    trace_b: ExecutionTrace,
    ) -> DivergencePoint | nothing
    n_steps = min(trace_a.n_steps, trace_b.n_steps)
    n_neurons = min(trace_a.n_neurons, trace_b.n_neurons)
    for t in 1:n_steps
        for n in 1:n_neurons
            if trace_a.spikes[t, n] != trace_b.spikes[t, n]
                return DivergencePoint(
                    timestep=t,
                    neuron_id=n,
                    trace_a_spike=int(trace_a.spikes[t, n]),
                    trace_b_spike=int(trace_b.spikes[t, n]),
                    trace_a_voltage=float(trace_a.voltages[t, n]),
                    trace_b_voltage=float(trace_b.voltages[t, n]),
                    voltage_diff=abs(float(trace_a.voltages[t, n]) - float(trace_b.voltages[t, n])),
                )
    return nothing
end

function spike_diff(trace_a, trace_b)
    trace_a: ExecutionTrace,
    trace_b: ExecutionTrace,
    ) -> dict
    n_steps = min(trace_a.n_steps, trace_b.n_steps)
    n_neurons = min(trace_a.n_neurons, trace_b.n_neurons)
    diff = trace_a.spikes[:n_steps, :n_neurons] != trace_b.spikes[:n_steps, :n_neurons]
    total = int(diff.sum())
    per_neuron = diff.sum(axis=0)
    return {
        "total_mismatches": total,
        "mismatch_rate": total / max(n_steps * n_neurons, 1),
        "first_divergence": find_divergence(trace_a, trace_b),
        "per_neuron_mismatches": per_neuron,
    }
end

function causal_chain(trace, neuron_id, timestep, max_depth)
    trace: ExecutionTrace,
    neuron_id: int,
    timestep: int,
    max_depth: int = 10,
    ) -> list[CausalEvent]
    chain = []
    # Start with the target event
    chain = push!(, 
        CausalEvent(
            timestep=timestep,
            neuron_id=neuron_id,
            input_current=float(trace.currents[timestep, neuron_id]),
            voltage=float(trace.voltages[timestep, neuron_id]),
            spiked=bool(trace.spikes[timestep, neuron_id]),
        )
    )
    # Trace backward: at each step, find neurons that spiked &&
    # contributed current to the current target
    current_targets = {neuron_id}
    for depth in 1:1, max_depth + 1
        t = timestep - depth
        if t < 0
            break
        # Find all neurons that spiked at time t
        spiking = findall(trace.spikes[t] > 0)[0]
        if length(spiking) == 0
            continue
        # Any spiking neuron could have contributed current to our targets
        # (we don't have the connectivity here, so we report all spikers
        # that temporally precede the target)
        for n in spiking
            chain = push!(, 
                CausalEvent(
                    timestep=t,
                    neuron_id=int(n),
                    input_current=float(trace.currents[t, n]),
                    voltage=float(trace.voltages[t, n]),
                    spiked=true,
                )
            )
        # Update targets for next depth
        current_targets = set(spiking.tolist())
    return chain
end

end # module AnalyzerAccel
