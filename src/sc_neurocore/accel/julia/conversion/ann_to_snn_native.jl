# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia rooted dense IF native ownership boundary

"""ABI-one native replay and rooted numeric buffer lifetime; no complete trace copy."""
module AnnToSnnNative

export sc_if_abi_version, sc_if_replay, sc_if_buffer, sc_if_free

include("ann_to_snn.jl")
include("ann_to_snn_native_types.jl")
include("ann_to_snn_native_request.jl")

const RESULTS = Dict{UInt,AnnToSnnAccel.ReplayResult}()
const NEXT_ID = Ref{UInt}(0)
const OWNER_LOCK = ReentrantLock()

"""Report the exact owned-buffer ABI version1."""
sc_if_abi_version()::UInt32 = 1

"""Replay borrowed C requests, publishing one owned result on success.

Returns0 success,-1 domain,-2 resource,-3 finite arithmetic overflow,-4 internal
failure. Refusal leaves the exclusive live aligned owner slot unchanged. All
nonempty pointers denote complete aligned live arrays unchanged during the call;
span checks do not prove OS accessibility. Julia calls require a registered
runtime thread. The Python adapter uses JuliaCall's managed Julia entry points.
"""
function sc_if_replay(request::Ptr{ReplayRequest}, result::Ptr{Ptr{Cvoid}})::Cint
    span(request, UInt(1)) && span(result, UInt(1)) || return -1
    slot = Ptr{UInt}(C_NULL)
    key = UInt(0)
    published = false
    try
        value = execute(unsafe_load(request))
        slot = Ptr{UInt}(Libc.malloc(sizeof(UInt)))
        slot != C_NULL || throw(OutOfMemoryError())
        lock(OWNER_LOCK) do
            NEXT_ID[] < typemax(UInt) || throw(OutOfMemoryError())
            key = NEXT_ID[] += 1
            RESULTS[key] = value
            unsafe_store!(slot, key)
            unsafe_store!(result, Ptr{Cvoid}(slot))
            published = true
        end
        return 0
    catch error
        key == 0 || lock(() -> pop!(RESULTS, key, nothing), OWNER_LOCK)
        return error isa ArgumentError ? -1 : error isa OutOfMemoryError ? -2 : error isa OverflowError ? -3 : -4
    finally
        published || slot == C_NULL || Libc.free(slot)
    end
end

"""Borrow one result vector: kind0 output/index0,1 final,2 state trace,3 IF events.

Unknown kind/index returns-1 without modifying the exclusive live aligned view.
Handle is a live unmodified owner from this module, never concurrently freed.
Returned doubles may be mutated; metadata stays unchanged. Rooted result vectors
remain live until exactly-once free. No raw C entry is safe on an unregistered
Julia runtime thread; Python uses managed JuliaCall entry points.
"""
function sc_if_buffer(handle::Ptr{UInt}, kind::UInt32, index::UInt, view::Ptr{BufferView})::Cint
    span(handle, UInt(1)) && span(view, UInt(1)) || return -1
    result = lock(() -> get(RESULTS, unsafe_load(handle), nothing), OWNER_LOCK)
    isnothing(result) && return -1
    groups = kind == 0 ? [result.output] : kind == 1 ? result.final_state :
             kind == 2 ? result.state_trace : kind == 3 ? result.spike_trace : nothing
    isnothing(groups) && return -1
    index < UInt(length(groups)) || return -1
    values = groups[Int(index)+1]
    GC.@preserve result values begin
        data = isempty(values) ? Ptr{Float64}(C_NULL) : pointer(values)
        unsafe_store!(view, BufferView(data, UInt(length(values))))
    end
    return 0
end

"""Drop the rooted result and free C metadata exactly once; null is harmless.

Nonnull handle must be live and from this module; every view must have expired
before release and cannot be accessed concurrently. Call from a registered Julia
runtime thread, using JuliaCall's managed entry point from Python.
"""
function sc_if_free(handle::Ptr{UInt})::Cvoid
    handle == C_NULL && return nothing
    lock(() -> pop!(RESULTS, unsafe_load(handle), nothing), OWNER_LOCK)
    Libc.free(handle)
    return nothing
end

"""JuliaCall managed pointer entry, preserving the C layout and refusal semantics."""
replay_pointer(request::Integer, result::Integer)::Int = Int(sc_if_replay(Ptr{ReplayRequest}(UInt(request)), Ptr{Ptr{Cvoid}}(UInt(result))))
"""JuliaCall managed borrowed-view entry; owner stays rooted until final release."""
buffer_pointer(handle::Integer, kind::Integer, index::Integer, view::Integer)::Int = Int(sc_if_buffer(Ptr{UInt}(UInt(handle)), UInt32(kind), UInt(index), Ptr{BufferView}(UInt(view))))
"""JuliaCall managed release entry; all views must have expired before this call."""
free_pointer(handle::Integer) = sc_if_free(Ptr{UInt}(UInt(handle)))

end # module AnnToSnnNative
