# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source-faithful McCulloch-Pitts Julia kernel

module McCullochPittsAccel

export McCullochPittsNeuronState, evaluate_batch, reset!, step!, validate_mcculloch_pitts

const INT32_MAX = Int64(2147483647)

mutable struct McCullochPittsNeuronState
    theta::Int64
    function McCullochPittsNeuronState(theta::Integer=1)
        theta64 = Int64(theta)
        if !(1 <= theta64 <= INT32_MAX)
            throw(DomainError(theta, "theta must be a positive signed 32-bit integer"))
        end
        return new(theta64)
    end
end

function validate_mcculloch_pitts(state::McCullochPittsNeuronState)::Bool
    return 1 <= state.theta <= INT32_MAX
end

function step!(
    state::McCullochPittsNeuronState,
    excitatory_count::Integer=0,
    inhibitory_active::Bool=false,
)::Int
    count = Int64(excitatory_count)
    if !validate_mcculloch_pitts(state)
        throw(DomainError(state.theta, "theta must be a positive signed 32-bit integer"))
    end
    if !(0 <= count <= INT32_MAX)
        throw(DomainError(
            excitatory_count,
            "excitatory_count must be a non-negative signed 32-bit integer",
        ))
    end
    return !inhibitory_active && count >= state.theta ? 1 : 0
end

function reset!(state::McCullochPittsNeuronState)::Nothing
    if !validate_mcculloch_pitts(state)
        throw(DomainError(state.theta, "theta must be a positive signed 32-bit integer"))
    end
    return nothing
end

function evaluate_batch(theta::Integer, excitatory_counts, inhibitory_flags)
    state = McCullochPittsNeuronState(theta)
    if length(excitatory_counts) != length(inhibitory_flags)
        throw(DimensionMismatch("inhibitory_flags must match excitatory_counts length"))
    end

    counts = Vector{Int64}(undef, length(excitatory_counts))
    flags = BitVector(undef, length(inhibitory_flags))
    for index in eachindex(excitatory_counts, inhibitory_flags)
        count = Int64(excitatory_counts[index])
        flag = Int64(inhibitory_flags[index])
        if !(0 <= count <= INT32_MAX)
            throw(DomainError(count, "excitatory counts must be in signed 32-bit range"))
        end
        if flag != 0 && flag != 1
            throw(DomainError(flag, "inhibitory flags must contain only zero or one"))
        end
        counts[index] = count
        flags[index] = flag == 1
    end

    events = Vector{UInt8}(undef, length(counts))
    event_count = Int64(0)
    for index in eachindex(counts)
        event = step!(state, counts[index], flags[index])
        events[index] = UInt8(event)
        event_count += event
    end
    return (events=events, event_count=event_count)
end

end # module McCullochPittsAccel
