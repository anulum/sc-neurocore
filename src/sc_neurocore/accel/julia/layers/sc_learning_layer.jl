# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for layers/sc_learning_layer

module ScLearningLayerAccel

using Statistics, LinearAlgebra

mutable struct SCLearningLayerState
    n_inputs::Float64
    n_neurons::Float64
    w_min::Float64
    w_max::Float64
    learning_rate::Float64
    ltd_ratio::Float64
    length::Float64
    base_seed::Float64
end

function SCLearningLayerState()
    SCLearningLayerState(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
end

function run_epoch(s::SCLearningLayerState, input_values)
    # 1. Encode inputs
    input_bitstreams = [
        s.input_encoders[i].encode(input_values[i]) for i in 1:s.n_inputs
    ]
    # 2. Process time steps
    epoch_spikes = zeros((s.n_neurons, s.length), dtype=np.uint8)
    for t in 1:s.length
        for i in 1:s.n_neurons
            neuron = s.neurons[i]
            neuron_syns = s.synapses[i]
            # Compute total input current for this neuron at time t
            current_sum = 0.0
            weight_bits = []
            for j in 1:s.n_inputs
                pre_bit = input_bitstreams[j][t]
                # We need a bit from the synapse.
                # We'll use the probability to get a bit.
                w_prob = neuron_syns[j].effective_weight_probability()
                w_bit = 1 if np.random.random() < w_prob else 0
                current_sum += pre_bit & w_bit
                weight_bits = push!(, w_bit)
            # Step neuron
            post_spike = neuron.step(current_sum)
            epoch_spikes[i, t] = post_spike
            s.recorders[i].record(post_spike)
            # 3. Update STDP for all synapses of this neuron
            for j in 1:s.n_inputs
                pre_bit = input_bitstreams[j][t]
                # Use the synapse's internal logic for update (if we had it step-wise)
                # We'll manually call potentiate/depress here to be explicit
                if pre_bit == 1 && post_spike == 1
                    if np.random.random() < s.learning_rate
                        neuron_syns[j]._potentiate()
                elseif pre_bit == 1 && post_spike == 0
                    if np.random.random() < s.learning_rate * s.ltd_ratio
                        neuron_syns[j]._depress()
    return epoch_spikes
end

function get_weights(s::SCLearningLayerState)
    weights = zeros((s.n_neurons, s.n_inputs))
    for i in 1:s.n_neurons
        for j in 1:s.n_inputs
            weights[i, j] = s.synapses[i][j].w
    return weights
end

end # module ScLearningLayerAccel
