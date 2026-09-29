# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia N-MNIST recording decoder

"""Decode complete Orchard et al. (2015) camera records without timestamp narrowing."""
module NMNISTRecordings

export decode_nmnist!, decode_nmnist, read_nmnist, load_nmnist, decode_nmnist_pointer

"""
    decode_nmnist!(output::Vector{Float64}, raw::Vector{UInt8})

Write row-major x, y, polarity and milliseconds into caller-owned output.
Validate both lengths before changing any destination value. Whole-byte
addresses and the 23-bit recorded microsecond timestamp retain their precision.
"""
function decode_nmnist!(output::Vector{Float64}, raw::Vector{UInt8})
    length(raw) % 5 == 0 || throw(ArgumentError("N-MNIST file contains an incomplete 40-bit event"))
    length(output) % 4 == 0 && length(output) ÷ 4 == length(raw) ÷ 5 ||
        throw(ArgumentError("N-MNIST output must have four columns per event"))
    for index in 0:(length(raw) ÷ 5 - 1)
        r = index * 5
        o = index * 4
        output[o + 1] = Float64(raw[r + 1])
        output[o + 2] = Float64(raw[r + 2])
        output[o + 3] = Float64(raw[r + 3] >> 7)
        time_us = (UInt32(raw[r + 3] & 0x7f) << 16) |
            (UInt32(raw[r + 4]) << 8) | UInt32(raw[r + 5])
        output[o + 4] = Float64(time_us) / 1000.0
    end
    return output
end

"""
    decode_nmnist(raw::Vector{UInt8}) -> Vector{Float64}

Allocate four row-major values per event; refuse an incomplete final record.
The result stores milliseconds independently of the downstream encoder's dt.
"""
function decode_nmnist(raw::Vector{UInt8})
    length(raw) % 5 == 0 || throw(ArgumentError("N-MNIST file contains an incomplete 40-bit event"))
    return decode_nmnist!(Vector{Float64}(undef, length(raw) ÷ 5 * 4), raw)
end

"""
    read_nmnist(path::AbstractString) -> Vector{Float64}

Read one binary recording, propagating I/O failures and refusing truncation.
"""
read_nmnist(path::AbstractString) = decode_nmnist(read(path))

"""
    load_nmnist(root::AbstractString; train::Bool=true)

Read the actual Train or Test split in sorted class/file order. Return vectors
of row-major recordings and Int64 labels. Follow directory symlinks; refuse
invalid class labels, unreadable recordings and truncated final events.
No download or synthetic substitution occurs.
"""
function load_nmnist(root::AbstractString; train::Bool=true)
    split = joinpath(root, train ? "Train" : "Test")
    samples = Vector{Float64}[]
    labels = Int64[]
    for name in readdir(split; sort=true)
        directory = joinpath(split, name)
        isdir(directory) || continue
        label = parse(Int64, name)
        for file in readdir(directory; sort=true)
            endswith(file, ".bin") || continue
            push!(samples, read_nmnist(joinpath(directory, file)))
            push!(labels, label)
        end
    end
    return samples, labels
end

"""
    decode_nmnist_pointer(raw_address, byte_count, output_address, value_count) -> Int32

Decode into borrowed row-major Float64 memory; return zero or -1 on refusal.
Lengths, null pointers, alignment, address overflow and overlap are checked
before constructing views or mutating output. Nonempty addresses must name live
allocations of the declared sizes. Input remains immutable and output exclusively
writable until the synchronous call returns; callers retain both allocations.
"""
function decode_nmnist_pointer(raw_address::Integer, byte_count::Integer,
                               output_address::Integer, value_count::Integer)
    maximum = typemax(Int)
    if !(0 <= byte_count <= maximum) || !(0 <= value_count <= maximum ÷ 8) ||
        byte_count % 5 != 0 || value_count % 4 != 0 || value_count ÷ 4 != byte_count ÷ 5
        return Int32(-1)
    end
    byte_count == 0 && return Int32(0)
    if !(0 < raw_address <= maximum - byte_count) ||
        !(0 < output_address <= maximum - value_count * 8) || output_address % 8 != 0
        return Int32(-1)
    end
    if raw_address < output_address + value_count * 8 && output_address < raw_address + byte_count
        return Int32(-1)
    end
    raw = unsafe_wrap(Vector{UInt8}, Ptr{UInt8}(UInt(raw_address)), Int(byte_count); own=false)
    output = unsafe_wrap(Vector{Float64}, Ptr{Float64}(UInt(output_address)), Int(value_count); own=false)
    decode_nmnist!(output, raw)
    return Int32(0)
end

end # module NMNISTRecordings
