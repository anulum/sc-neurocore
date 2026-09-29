# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia indexed SHD recordings

"""Read indexed auditory recordings directly through system HDF5."""
module SHDRecordings

include("shd_hdf5.jl")
using .SHDHDF5

export read_shd_recording

const HDF5_LOCK = ReentrantLock()

"""Read selected paired vectors, label and event matrix under the caller-owned context."""
function read_row(context::SHDHDF5.Context, path::AbstractString, index::Int, maximum::Int)
    ccall(SHDHDF5.symbol(context, :H5open), Cint, ()) >= 0 || error("HDF5 initialisation failed")
    file = SHDHDF5.own!(context, ccall(SHDHDF5.symbol(context, :H5Fopen), Int64,
                                    (Cstring,Cuint,Int64), path, 0, 0), :H5Fclose)
    times = SHDHDF5.dataset(context, file, "spikes/times")
    units = SHDHDF5.dataset(context, file, "spikes/units")
    labels = SHDHDF5.dataset(context, file, "labels")
    SHDHDF5.numeric_vlen(context, times)
    SHDHDF5.numeric_vlen(context, units)
    ts, tl = SHDHDF5.selected_space(context, times, index)
    us, ul = SHDHDF5.selected_space(context, units, index)
    ls, ll = SHDHDF5.selected_space(context, labels, index)
    tl == ul == ll || error("SHD recording counts differ")
    one = Ref{UInt64}(1)
    memory = SHDHDF5.own!(context, ccall(SHDHDF5.symbol(context, :H5Screate_simple), Int64,
                                       (Cint,Ptr{UInt64},Ptr{UInt64}), 1, one, C_NULL), :H5Sclose)
    native = unsafe_load(Ptr{Int64}(SHDHDF5.symbol(context, :H5T_NATIVE_DOUBLE_g)))
    type = SHDHDF5.own!(context, ccall(SHDHDF5.symbol(context, :H5Tvlen_create), Int64,
                                     (Int64,), native), :H5Tclose)
    tb = SHDHDF5.vector_bytes(context, times, type, ts)
    ub = SHDHDF5.vector_bytes(context, units, type, us)
    tb <= UInt64(maximum ÷ 2) && ub <= UInt64(maximum ÷ 2) - tb ||
        error("SHD recording exceeds its event budget")
    time = SHDHDF5.vector(context, times, type, memory, ts, maximum)
    channels = SHDHDF5.vector(context, units, type, memory, us, maximum)
    length(time) == length(channels) || error("SHD event vectors differ")
    label = SHDHDF5.label(context, labels, memory, ls)
    events = zeros(Float64, length(time) * 4)
    for i in eachindex(time)
        events[4i - 3] = channels[i]
        events[4i] = time[i] * 1000.0
    end
    return (events=events, label=label)
end

"""
    read_shd_recording(path, index; maximum_bytes=64*1024*1024,
                       hdf5_library="libhdf5_serial.so")

Read one zero-based HDF5 recording as row-major Float64 x/y/polarity/t_ms values
and an Int64 label. Numeric variable-length time/channel vectors and integer
labels must have matching rank-one recording counts. Widen seconds before
multiplication; preserve empty recordings. Refuse malformed data, index, label
overflow, budget or missing library without download or synthetic substitution.
The event-result byte budget does not cap HDF5 buffers or transient copies.
The caller verifies manifest identity, geometry and finiteness separately.
All owned identifiers and variable-length buffers are closed or reclaimed before
returning. Use a separate Julia process to avoid foreign library interposition;
calls are serialised within this reader, independently of foreign HDF5 callers.
"""
function read_shd_recording(path::AbstractString, index::Integer;
                            maximum_bytes::Integer=64*1024*1024,
                            hdf5_library::AbstractString="libhdf5_serial.so")
    !isempty(path) && !occursin('\0', path) || throw(ArgumentError("invalid SHD path"))
    !(index isa Bool) && 0 <= index <= typemax(Int) || throw(ArgumentError("invalid SHD index"))
    !(maximum_bytes isa Bool) && 0 <= maximum_bytes <= typemax(Int) ||
        throw(ArgumentError("invalid SHD event budget"))
    return lock(HDF5_LOCK) do
        context = SHDHDF5.open_context(hdf5_library)
        try
            read_row(context, path, Int(index), Int(maximum_bytes))
        finally
            SHDHDF5.close!(context)
        end
    end
end

end
