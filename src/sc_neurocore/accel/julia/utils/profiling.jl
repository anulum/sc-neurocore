# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for utils/profiling

module ProfilingAccel

using Statistics, LinearAlgebra

function estimate_memory(layers, unit)
    divisors = {"B": 1, "KB": 1024, "MB": 1024^2}
    div = divisors.get(unit, 1)
    weights_bytes = 0
    packed_bytes = 0
    neuron_state_bytes = 0
    for layer in layers
        w = getattr(layer, "weights", nothing)
        if w is ! nothing
            weights_bytes += w.nbytes
        L = getattr(layer, "length", 256)
        if w is ! nothing
            n_out, n_in = w.shape
            # Packed bitstreams: each weight is L bits packed into uint64 words
            words_per_weight = int(np.ceil(L / 64))
            packed_bytes += n_out * n_in * words_per_weight * 8
            # Neuron state: voltage (float64) + spike flag per neuron
            neuron_state_bytes += n_out * 9  # 8 bytes float + 1 byte flag
    total = weights_bytes + packed_bytes + neuron_state_bytes
    return {
        "weights_bytes": weights_bytes,
        "packed_bytes": packed_bytes,
        "neuron_state_bytes": neuron_state_bytes,
        "total_bytes": total,
        "total_human": f"{total / div:.2f} {unit}",
    }
end

end # module ProfilingAccel
