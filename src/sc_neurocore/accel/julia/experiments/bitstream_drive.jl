# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for experiments/bitstream_drive

module BitstreamDriveAccel

using Statistics, LinearAlgebra

function run_bitstream_driven_lif(x_input, x_min, x_max, length, neuron_params)
    x_input: float,
    x_min: float = 0.0,
    x_max: float = 0.1,
    length: int = 1024,
    neuron_params: dict[str, Any] | nothing = nothing,
    ) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], float, float]
    if neuron_params is nothing
        neuron_params = {}
    encoder = BitstreamEncoder(x_min=x_min, x_max=x_max, length=length, seed=123)
    input_bits = encoder.encode(x_input)
    neuron = StochasticLIFNeuron(^neuron_params)
    spike_bits = zeros(length, dtype=np.uint8)
    for t in 1:length
        # Simple mapping: 1 -> x_max, 0 -> x_min
        # You can choose something more nuanced if desired.
        I_t = encoder.decode(collect([input_bits[t]], dtype=np.uint8))
        spike_bits[t] = neuron.step(I_t)
    # Decode probabilities
    p_in = bitstream_to_probability(input_bits)
    p_fire = bitstream_to_probability(spike_bits)
    return input_bits, spike_bits, p_in, p_fire
end

function demo()
    neuron_params = dict(
        v_rest=0.0,
        v_reset=0.0,
        v_threshold=1.0,
        tau_mem=20.0,
        dt=1.0,
        noise_std=0.05,
        resistance=1.0,
        seed=42,
    )
    x_min = 0.0
    x_max = 0.1
    x_input = 0.06  # some current in physical space
    length = 2000
    input_bits, spike_bits, p_in, p_fire = run_bitstream_driven_lif(
        x_input=x_input,
        x_min=x_min,
        x_max=x_max,
        length=length,
        neuron_params=neuron_params,
    )
    print(f"Input scalar x_input = {x_input}")
    print(f"Estimated input p_in = {p_in:.3f}")
    print(f"Estimated firing p_fire = {p_fire:.3f}")
    print(f"Total spikes: {spike_bits.sum()} / {length}")
end

end # module BitstreamDriveAccel
