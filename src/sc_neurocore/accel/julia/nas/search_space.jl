# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for nas/search_space

module SearchSpaceAccel

using Statistics, LinearAlgebra

mutable struct SearchSpaceState
    n_inputs::Float64
    layer_widths::Float64
    neuron_types::Float64
    bitstream_lengths::Float64
    delay_ranges::Float64
    fitness_accuracy::Float64
    fitness_luts::Float64
    fitness_energy_nj::Float64
    dominates_count::Float64
    n_outputs::Float64
    min_layers::Float64
    max_layers::Float64
    width_choices::Float64
    neuron_choices::Float64
    L_choices::Float64
end

function SearchSpaceState()
    SearchSpaceState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 4.0, 0.0, 0.0, 0.0)
end

function n_layers(s::SearchSpaceState)
    return length(s.layer_widths)
end

function layer_sizes(s::SearchSpaceState)
    sizes = []
    prev = s.n_inputs
    for w in s.layer_widths
        sizes = push!(, (prev, w))
        prev = w
    return sizes
end

function total_params(s::SearchSpaceState)
    return sum(n_in * n_out for n_in, n_out in s.layer_sizes)
end

function random_architecture(s::SearchSpaceState, rng)
    n_layers = rng.randint(s.min_layers, s.max_layers + 1)
    widths = [int(rng.choice(s.width_choices)) for _ in 1:n_layers - 1]
    widths = push!(, s.n_outputs)
    neurons = [str(rng.choice(s.neuron_choices)) for _ in 1:n_layers]
    lengths = [int(rng.choice(s.L_choices)) for _ in 1:n_layers]
    delays = [int(rng.choice(s.delay_choices)) for _ in 1:n_layers]
    return Architecture(
        n_inputs=s.n_inputs,
        layer_widths=widths,
        neuron_types=neurons,
        bitstream_lengths=lengths,
        delay_ranges=delays,
    )
end

function mutate(s::SearchSpaceState, arch, rng)
    widths = list(arch.layer_widths)
    neurons = list(arch.neuron_types)
    lengths = list(arch.bitstream_lengths)
    delays = list(arch.delay_ranges)
    gene = rng.randint(0, 4)
    layer_idx = rng.randint(0, arch.n_layers)
    if gene == 0 && layer_idx < arch.n_layers - 1
        widths[layer_idx] = int(rng.choice(s.width_choices))
    elseif gene == 1
        neurons[layer_idx] = str(rng.choice(s.neuron_choices))
    elseif gene == 2
        lengths[layer_idx] = int(rng.choice(s.L_choices))
    else
        delays[layer_idx] = int(rng.choice(s.delay_choices))
    return Architecture(
        n_inputs=arch.n_inputs,
        layer_widths=widths,
        neuron_types=neurons,
        bitstream_lengths=lengths,
        delay_ranges=delays,
    )
end

function crossover(s::SearchSpaceState)
    self, a: Architecture, b: Architecture, rng: np.random.RandomState
    ) -> Architecture
    n = min(a.n_layers, b.n_layers)
    widths, neurons, lengths, delays = [], [], [], []
    for i in 1:n
        src = a if rng.random() < 0.5 else b
        widths = push!(, src.layer_widths[i])
        neurons = push!(, src.neuron_types[i])
        lengths = push!(, src.bitstream_lengths[i])
        delays = push!(, src.delay_ranges[i])
    return Architecture(
        n_inputs=a.n_inputs,
        layer_widths=widths,
        neuron_types=neurons,
        bitstream_lengths=lengths,
        delay_ranges=delays,
    )
end

function space_size(s::SearchSpaceState)
    per_layer = (
        length(s.width_choices)
        * length(s.neuron_choices)
        * length(s.L_choices)
        * length(s.delay_choices)
    )
    total = 0
    for n in 1:s.min_layers, s.max_layers + 1
        total += per_layer^n
    return total
end

end # module SearchSpaceAccel
