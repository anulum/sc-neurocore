# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for synapses/sc_synapse

module ScSynapseAccel

using Statistics, LinearAlgebra

mutable struct BitstreamSynapseState
    w_min::Float64
    w_max::Float64
    length::Float64
    w::Float64
    seed::Float64
end

function BitstreamSynapseState()
    BitstreamSynapseState(0.0, 0.0, 0.0, 0.0, 0.0)
end

function encode_weight(s::BitstreamSynapseState, w)
    return s._weight_encoder.encode(w)
end

function update_weight(s::BitstreamSynapseState, new_w)
    s.w = new_w
    s.weight_bits = s.encode_weight(new_w)
end

function apply(s::BitstreamSynapseState, pre_bits, Any])
    if pre_bits.shape[0] != s.weight_bits.shape[0]
        raise ValueError(
            f"Bitstream length mismatch: pre={pre_bits.shape[0]}, "
            f"weight={s.weight_bits.shape[0]}"
        )
    # Logical AND implements multiplication in SC domain
    result: np.ndarray[Any, Any] = (pre_bits & s.weight_bits).astype(np.uint8)
    return result
end

function effective_weight_probability(s::BitstreamSynapseState)
    return bitstream_to_probability(s.weight_bits)
end

end # module ScSynapseAccel
