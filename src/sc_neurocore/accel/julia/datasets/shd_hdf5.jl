# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia HDF5 resource ownership

"""Own the identifiers and variable-length allocations of one system HDF5 read."""
module SHDHDF5

using Libdl

"""Own the loaded HDF5 handle and its successful object opens in acquisition order."""
mutable struct Context
    library::Ptr{Cvoid}
    objects::Vector{Tuple{Int64,Ptr{Cvoid}}}
end

"""Match the HDF5 native hvl_t size and double-vector pointer layout."""
struct Vlen
    count::Csize_t
    data::Ptr{Float64}
end

"""Reserve the known resource stack before loading the operator-selected HDF5 library."""
function open_context(library::AbstractString)
    objects = Tuple{Int64,Ptr{Cvoid}}[]
    sizehint!(objects, 32)
    return Context(dlopen(library), objects)
end

"""Resolve one symbol from the owned library handle."""
symbol(context::Context, name::Symbol) = dlsym(context.library, name)

"""Register a successful HDF5 identifier with its matching close function."""
function own!(context::Context, id::Int64, closer::Symbol)
    id >= 0 || error("HDF5 object is unavailable")
    push!(context.objects, (id, symbol(context, closer)))
    return id
end

"""Close every owned identifier in reverse order and unload the library; propagate close failure."""
function close!(context::Context)
    failed = false
    try
        for (id, closer) in Iterators.reverse(context.objects)
            failed |= ccall(closer, Cint, (Int64,), id) < 0
        end
    finally
        empty!(context.objects)
        dlclose(context.library)
    end
    failed && error("HDF5 identifier close failed")
    return nothing
end

"""Open one named dataset from the already-owned read-only HDF5 file."""
function dataset(context::Context, file::Int64, name::AbstractString)
    id = ccall(symbol(context, :H5Dopen2), Int64, (Int64,Cstring,Int64), file, name, 0)
    return own!(context, id, :H5Dclose)
end

"""Own one dataset datatype until the context closes."""
function datatype(context::Context, dataset::Int64)
    return own!(context, ccall(symbol(context, :H5Dget_type), Int64, (Int64,), dataset), :H5Tclose)
end

"""Select one zero-based recording and return its rank-one dataspace and total row count."""
function selected_space(context::Context, dataset::Int64, index::Int)
    space = own!(context, ccall(symbol(context, :H5Dget_space), Int64, (Int64,), dataset), :H5Sclose)
    dimensions = Ref{UInt64}(0)
    start, count = Ref{UInt64}(index), Ref{UInt64}(1)
    ccall(symbol(context, :H5Sget_simple_extent_ndims), Cint, (Int64,), space) == 1 ||
        error("SHD datasets must have rank one")
    ccall(symbol(context, :H5Sget_simple_extent_dims), Cint,
          (Int64,Ptr{UInt64},Ptr{UInt64}), space, dimensions, C_NULL) >= 0 ||
        error("HDF5 extent read failed")
    UInt64(index) < dimensions[] || throw(ArgumentError("SHD recording index is out of range"))
    ccall(symbol(context, :H5Sselect_hyperslab), Cint,
          (Int64,Cint,Ptr{UInt64},Ptr{UInt64},Ptr{UInt64},Ptr{UInt64}),
          space, 0, start, C_NULL, count, C_NULL) >= 0 || error("HDF5 row selection failed")
    return space, dimensions[]
end

"""Refuse any event dataset without a numeric variable-length base datatype."""
function numeric_vlen(context::Context, dataset::Int64)
    type = datatype(context, dataset)
    ccall(symbol(context, :H5Tget_class), Cint, (Int64,), type) == 9 ||
        error("SHD events must be variable-length vectors")
    base = own!(context, ccall(symbol(context, :H5Tget_super), Int64, (Int64,), type), :H5Tclose)
    ccall(symbol(context, :H5Tget_class), Cint, (Int64,), base) in (0,1) ||
        error("SHD events must be numeric")
    return nothing
end

"""Estimate the selected native-double vector allocation before its explicit read."""
function vector_bytes(context::Context, dataset::Int64, type::Int64, space::Int64)
    bytes = Ref{UInt64}(0)
    ccall(symbol(context, :H5Dvlen_get_buf_size), Cint,
          (Int64,Int64,Int64,Ptr{UInt64}), dataset, type, space, bytes) >= 0 ||
        error("HDF5 vector allocation estimate failed")
    return bytes[]
end

"""Read and copy one bounded native-double vector, always reclaiming its HDF5 allocation."""
function vector(context::Context, dataset::Int64, type::Int64, memory::Int64, space::Int64, maximum::Int)
    value = Ref(Vlen(0, C_NULL))
    try
        ccall(symbol(context, :H5Dread), Cint,
              (Int64,Int64,Int64,Int64,Int64,Ptr{Cvoid}),
              dataset, type, memory, space, 0, value) >= 0 || error("HDF5 vector read failed")
        value[].count <= UInt64(maximum ÷ 32) || error("SHD recording exceeds its event budget")
        value[].count == 0 && return Float64[]
        value[].data != C_NULL || error("HDF5 returned an absent vector")
        return copy(unsafe_wrap(Vector{Float64}, value[].data, Int(value[].count); own=false))
    finally
        ccall(symbol(context, :H5Dvlen_reclaim), Cint,
              (Int64,Int64,Int64,Ptr{Cvoid}), type, memory, 0, value) >= 0 ||
            error("HDF5 vector reclaim failed")
    end
end

"""Read an integer label without saturating unsigned values outside int64."""
function label(context::Context, dataset::Int64, memory::Int64, space::Int64)
    type = datatype(context, dataset)
    ccall(symbol(context, :H5Tget_class), Cint, (Int64,), type) == 0 &&
        ccall(symbol(context, :H5Tget_size), Csize_t, (Int64,), type) <= 8 ||
        error("SHD labels must be representable integers")
    sign = ccall(symbol(context, :H5Tget_sign), Cint, (Int64,), type)
    sign in (0,1) || error("HDF5 label sign read failed")
    if sign == 0
        native = unsafe_load(Ptr{Int64}(symbol(context, :H5T_NATIVE_UINT64_g)))
        value = Ref{UInt64}(0)
        ccall(symbol(context, :H5Dread), Cint,
              (Int64,Int64,Int64,Int64,Int64,Ptr{Cvoid}),
              dataset, native, memory, space, 0, value) >= 0 || error("HDF5 label read failed")
        value[] <= typemax(Int64) || error("SHD label exceeds int64")
        return Int64(value[])
    end
    native = unsafe_load(Ptr{Int64}(symbol(context, :H5T_NATIVE_INT64_g)))
    value = Ref{Int64}(0)
    ccall(symbol(context, :H5Dread), Cint,
          (Int64,Int64,Int64,Int64,Int64,Ptr{Cvoid}),
          dataset, native, memory, space, 0, value) >= 0 || error("HDF5 label read failed")
    return value[]
end

end
