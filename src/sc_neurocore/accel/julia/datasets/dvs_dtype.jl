# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia converted DVS recording decoder

"""Validated real scalar width and source byte order."""
struct DVSDtype
    kind::Char
    width::Int
    big::Bool
end
"""Resolve NumPy real scalar descriptors, named aliases and type-number characters."""
function dtype(source::String)::DVSDtype
    isempty(source) && throw(ArgumentError("empty DVS dtype"))
    explicit = first(source) in ('<','>','=','|')
    prefix = explicit ? first(source) : '='
    word = explicit ? source[nextind(source,firstindex(source)):end] : source
    aliases = Dict("bool"=>"b1","bool_"=>"b1","byte"=>"i1","ubyte"=>"u1","short"=>"i2","ushort"=>"u2","intc"=>"i4","uintc"=>"u4","longlong"=>"i8","ulonglong"=>"u8","half"=>"f2","single"=>"f4","double"=>"f8","float"=>"f8","longdouble"=>"f16","float128"=>"f16")
    for (name,kind) in (("int",'i'),("uint",'u'),("float",'f'))
        for bits in (8,16,32,64)
            name == "float" && bits == 8 && continue
            aliases[name*string(bits)] = string(kind,bits÷8)
        end
    end
    if explicit && (haskey(aliases,word) || word in ("int","int_","intp","uint","uintp","long","ulong"))
        throw(ArgumentError("prefixed named DVS dtype"))
    end
    character = Dict("?"=>"b1","b"=>"i1","B"=>"u1","h"=>"i2","H"=>"u2","i"=>"i4","I"=>"u4","q"=>"i8","Q"=>"u8","e"=>"f2","f"=>"f4","d"=>"f8","g"=>"f16")
    for (index,value) in enumerate(("b1","i1","u1","i2","u2","i4","u4","i"*string(sizeof(Clong)),"u"*string(sizeof(Clong)),"i8","u8","f4","f8","f16"))
        character[string(Char(index-1))] = value
    end
    character[string(Char(23))] = "f2"
    for key in ("int","int_","intp","p","n")
        character[key] = "i"*string(sizeof(Int))
    end
    for key in ("uint","uintp","P","N")
        character[key] = "u"*string(sizeof(UInt))
    end
    character["long"] = character["l"] = "i"*string(sizeof(Clong))
    character["ulong"] = character["L"] = "u"*string(sizeof(Culong))
    resolved = get(character,word,get(aliases,word,word))
    isascii(resolved) && ncodeunits(resolved) >= 2 || throw(ArgumentError("invalid DVS dtype"))
    kind = first(resolved);width = parse(Int,resolved[2:end])
    (kind == 'b' && width == 1 || kind in ('i','u') && width in (1,2,4,8) || kind == 'f' && width in (2,4,8,16)) || throw(ArgumentError("nonreal DVS dtype"))
    width == 16 && !(Sys.islinux() && Sys.ARCH == :x86_64) && throw(ArgumentError("DVS extended encoding requires Linux x86-64"))
    big = prefix == '>' || prefix != '<' && ENDIAN_BOM == 0x01020304
    return DVSDtype(kind,width,big)
end
"""Read one at-most-eight-byte scalar without alignment assumptions."""
function scalar_bits(raw::Vector{UInt8},offset::Int,width::Int,big::Bool)::UInt64
    bits = UInt64(0)
    for index in 0:width-1
        shift = (big ? width-1-index : index)*8
        bits |= UInt64(raw[offset+index]) << shift
    end
    return bits
end
"""Convert the host-qualified padded x87 encoding with one final float64 rounding."""
function extended_scalar(raw::Vector{UInt8},offset::Int,big::Bool)::Float64
    significand = scalar_bits(raw,offset+(big ? 8 : 0),8,big)
    exponent = scalar_bits(raw,offset+(big ? 6 : 8),2,big)
    negative = exponent & 0x8000 != 0
    exponent &= 0x7fff
    value = if exponent == 0x7fff
        significand == UInt64(1)<<63 ? Inf : NaN
    elseif exponent != 0 && significand & (UInt64(1)<<63) == 0
        NaN
    else
        power = Int(max(exponent,1))-16383-63
        Float64(ldexp(BigFloat(significand;precision=64),power),RoundNearest)
    end
    return negative ? -value : value
end
"""Convert one admitted stored scalar into an independent float64 value."""
function scalar(raw::Vector{UInt8},offset::Int,d::DVSDtype)::Float64
    d.kind == 'b' && return raw[offset] == 0 ? 0.0 : 1.0
    d.width == 16 && return extended_scalar(raw,offset,d.big)
    bits = scalar_bits(raw,offset,d.width,d.big)
    if d.kind == 'u'
        return Float64(bits)
    elseif d.kind == 'i'
        value = d.width == 1 ? reinterpret(Int8,UInt8(bits)) : d.width == 2 ? reinterpret(Int16,UInt16(bits)) : d.width == 4 ? reinterpret(Int32,UInt32(bits)) : reinterpret(Int64,bits)
        return Float64(value)
    end
    return d.width == 2 ? Float64(reinterpret(Float16,UInt16(bits))) : d.width == 4 ? Float64(reinterpret(Float32,UInt32(bits))) : reinterpret(Float64,bits)
end
