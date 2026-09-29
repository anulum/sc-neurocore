# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia dense IF ABI-one borrowed descriptors

"""ABI-one C layout; complete aligned borrowed dense coefficients/state stay live during replay."""
struct LayerSpec
    outputs::Csize_t
    inputs::Csize_t
    weights::Ptr{Float64}
    bias::Ptr{Float64}
    bias_len::Csize_t
    threshold::Float64
    initial_fraction::Float64
    initial::Ptr{Float64}
    initial_len::Csize_t
end

"""ABI-one request: version1; flags1 trace,2 binary,4 linear final,8 initial state."""
struct ReplayRequest
    version::UInt32
    flags::UInt32
    layers::Ptr{LayerSpec}
    layer_count::Csize_t
    frames::Ptr{Float64}
    frames_len::Csize_t
    steps::Csize_t
    batch::Csize_t
    max_working_bytes::Csize_t
end

"""Borrowed row-major doubles live until the supplying opaque result owner is freed."""
struct BufferView
    data::Ptr{Float64}
    len::Csize_t
end

"""Admit aligned complete spans in the signed host address domain; zero count borrows nothing."""
function span(pointer::Ptr{T}, count::UInt) where T
    count == 0 && return true
    maximum = UInt(typemax(Int))
    address = UInt(pointer)
    width = UInt(sizeof(T))
    return address != 0 && address % UInt(Base.datatype_alignment(T)) == 0 &&
           count <= maximum ÷ width && address <= maximum - count * width
end

"""Borrow live complete caller storage without owning it; avoid Julia null vectors at zero count."""
function borrowed(pointer::Ptr{T}, count::UInt) where T
    span(pointer, count) || throw(ArgumentError("invalid native buffer span"))
    return count == 0 ? T[] : unsafe_wrap(Vector{T}, pointer, Int(count); own=false)
end
