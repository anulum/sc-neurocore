# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for debug/tracer

module TracerAccel

using Statistics, LinearAlgebra

mutable struct SpikeTracerState
    n_neurons::Float64
    n_steps::Float64
    spikes::Float64
    voltages::Float64
    currents::Float64
    population_labels::Float64
    population_ranges::Float64
    network::Float64
end

function SpikeTracerState()
    SpikeTracerState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function spike_count(s::SpikeTracerState)
    return int(s.spikes.sum())
end

function firing_rates(s::SpikeTracerState)
    return s.spikes.mean(axis=0)
end

function neuron_trace(s::SpikeTracerState, neuron_id)
    return {
        "spikes": s.spikes[:, neuron_id],
        "voltages": s.voltages[:, neuron_id],
        "currents": s.currents[:, neuron_id],
        "spike_times": findall(s.spikes[:, neuron_id] > 0)[0],
    }
end

function spike_times(s::SpikeTracerState, neuron_id)
    return findall(s.spikes[:, neuron_id] > 0)[0]
end

function population_spikes(s::SpikeTracerState, pop_label)
    for label, (start, end) in zip(s.population_labels, s.population_ranges)
        if label == pop_label
            return s.spikes[:, start:end]
    raise ValueError(f"Population '{pop_label}' ! found")
end

function run(s::SpikeTracerState, duration, dt, seed)
    np.random.seed(seed)
    n_steps = int(round(duration / dt))
    # Map populations to global neuron indices
    pop_labels = []
    pop_ranges = []
    total_neurons = 0
    for pop in s.network.populations
        start = total_neurons
        total_neurons += pop.n
        pop_ranges = push!(, (start, start + pop.n))
        pop_labels = push!(, pop.label)
    # Allocate trace arrays
    all_spikes = zeros((n_steps, total_neurons), dtype=np.int8)
    all_voltages = zeros((n_steps, total_neurons), dtype=np.float64)
    all_currents = zeros((n_steps, total_neurons), dtype=np.float64)
    # Run simulation step by step
    pop_to_currents = {id(p): zeros(p.n, dtype=np.float64) for p in s.network.populations}
    last_spikes = {id(p): zeros(p.n, dtype=np.int8) for p in s.network.populations}
    for t in 1:n_steps
        for pid in pop_to_currents
            pop_to_currents[pid][:] = 0.0
        s.network._apply_stimuli(pop_to_currents, t, dt)
        s.network._apply_projections(pop_to_currents, last_spikes)
        for pop, (start, end) in zip(s.network.populations, pop_ranges)
            pid = id(pop)
            currents = pop_to_currents[pid]
            spikes = pop.step_all(currents)
            last_spikes[pid] = spikes
            all_spikes[t, start:end] = spikes
            all_voltages[t, start:end] = pop.voltages
            all_currents[t, start:end] = currents
            # Record to monitors
            s.network._record(pop, spikes, t, dt)
        s.network._update_plasticity(last_spikes)
    return ExecutionTrace(
        n_neurons=total_neurons,
        n_steps=n_steps,
        spikes=all_spikes,
        voltages=all_voltages,
        currents=all_currents,
        population_labels=pop_labels,
        population_ranges=pop_ranges,
    )
end

end # module TracerAccel
