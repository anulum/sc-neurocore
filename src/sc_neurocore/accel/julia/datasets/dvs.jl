# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia converted DVS recording decoder

"""Read bounded converted DVS NPY recordings without Python or synthetic substitution."""
module DVSRecordings
export read_dvs_recording
include("dvs_literals.jl")
include("dvs_strings.jl")
include("dvs_dtype.jl")
"""Validate one unique three-field metadata dictionary and four-column event shape."""
function header(text::String)::Tuple{Int,Bool,DVSDtype}
    occursin('\0',text) && throw(ArgumentError("NUL in DVS header"))
    normalized = replace(replace(text,"\r\n"=>"\n"),"\r"=>"\n")
    parser = DVSParser(collect(codeunits(normalized)),1,0)
    whitespace!(parser)
    prefix = String(parser.text[1:parser.position-1])
    indentation = last(split(last(split(prefix,'\n')),'\f'))
    !isempty(indentation) && all(c->c in (' ','\t'),indentation) && throw(ArgumentError("indented DVS header"))
    root = literal!(parser);whitespace!(parser)
    parser.position == length(parser.text)+1 || throw(ArgumentError("trailing DVS expression"))
    root isa DVSDictionary && Set(keys(root.values)) == Set(("descr","fortran_order","shape")) || throw(ArgumentError("invalid DVS header fields"))
    descriptor = root.values["descr"];order = root.values["fortran_order"];shape = root.values["shape"]
    descriptor isa DVSString && order isa DVSBoolean && shape isa DVSTuple && length(shape.values) == 2 || throw(ArgumentError("invalid DVS metadata types"))
    rows,columns = shape.values
    rows isa DVSNumber && columns isa DVSNumber && 0 <= rows.value <= typemax(Int) && columns.value == 4 || throw(ArgumentError("invalid DVS event shape"))
    return Int(rows.value),order.value,dtype(descriptor.value)
end
"""Read exactly the declared bytes, refusing truncation."""
function exact(file::IO,count::Int)::Vector{UInt8}
    value = read(file,count)
    length(value) == count || throw(ArgumentError("incomplete DVS recording"))
    return value
end
"""Decode a borrowed stream under its validated budget, without owning its lifetime."""
Base.@noinline function decode_array(file::IO, maximum_bytes::Int)::Matrix{Float64}
    prefix = exact(file,8)
    prefix[1:6] == UInt8[0x93,0x4e,0x55,0x4d,0x50,0x59] && prefix[7] in (1,2,3) && prefix[8] == 0 || throw(ArgumentError("invalid DVS NPY preamble"))
    length_width = prefix[7] == 1 ? 2 : 4
    length_bytes = exact(file,length_width)
    count_header = Int(scalar_bits(length_bytes,1,length_width,false))
    0 < count_header <= 10000 || throw(ArgumentError("invalid DVS header length"))
    raw_header = exact(file,count_header)
    last(raw_header) == 0x0a || throw(ArgumentError("DVS header requires final newline"))
    text = prefix[7] == 3 ? String(raw_header) : join(Char.(raw_header))
    isvalid(text) || throw(ArgumentError("invalid DVS UTF8 header"))
    rows,fortran,d = header(text)
    rows <= maximum_bytes÷32 || throw(ArgumentError("DVS event budget exceeded"))
    count = rows*4
    count <= typemax(Int)÷d.width || throw(ArgumentError("DVS source payload exceeds native size"))
    payload = exact(file,count*d.width)
    eof(file) || throw(ArgumentError("DVS recording has extra content"))
    result = Matrix{Float64}(undef,rows,4)
    for index in 0:count-1
        row,column = fortran ? (index%rows+1,index÷rows+1) : (index÷4+1,index%4+1)
        result[row,column] = scalar(payload,index*d.width+1,d)
    end
    result
end
"""
    read_dvs_recording(path::AbstractString; maximum_bytes=64*1024*1024) -> Matrix{Float64}

Read one real numeric NPY array (versions 1.0/2.0/3.0) with x, y, polarity and
stored millisecond timestamp columns. C/Fortran storage and either byte order
preserve values through float64 conversion. Return independent writable events;
refuse invalid budgets, headers, shapes, truncation and extra content. The result
budget excludes source and temporary buffers. Extended encodings require a Linux
x86-64 host. Geometry, finiteness, time ordering and manifest identity are caller
contracts; no download, pickle, evaluation or synthetic replacement occurs.
A compiler boundary keeps decoder inference after file acquisition; the file
is closed on every return or error.
"""
Base.@noinline function read_dvs_recording(path::AbstractString;maximum_bytes::Integer=64*1024*1024)::Matrix{Float64}
    !(maximum_bytes isa Bool) && 0 <= maximum_bytes <= typemax(Int) || throw(ArgumentError("invalid DVS budget"))
    file = open(path,"r")
    try
        decoder = Base.inferencebarrier(decode_array)
        return decoder(file,Int(maximum_bytes))::Matrix{Float64}
    finally
        close(file)
    end
end
end # module DVSRecordings
