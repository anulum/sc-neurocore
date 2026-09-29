# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia converted DVS recording decoder

"""Represent only inert Python header literals; no evaluation or calls are permitted."""
abstract type DVSLiteral end
"""A decoded Python string literal with preserved scalar descriptor text."""
struct DVSString <: DVSLiteral
    value::String
end
"""An arbitrary-precision integer retaining whether a unary sign was applied."""
struct DVSNumber <: DVSLiteral
    value::BigInt
    signed::Bool
end
"""A Boolean literal kept distinct from integer dimensions."""
struct DVSBoolean <: DVSLiteral
    value::Bool
end
"""An ordered tuple of inert values, distinct from parenthesis grouping."""
struct DVSTuple <: DVSLiteral
    values::Vector{DVSLiteral}
end
"""A dictionary whose string keys were checked for duplicates."""
struct DVSDictionary <: DVSLiteral
    values::Dict{String,DVSLiteral}
end
"""Byte position and structural depth within one bounded UTF8 header."""
mutable struct DVSParser
    text::Vector{UInt8}
    position::Int
    depth::Int
end
"""Return an ASCII token or the explicit end-of-header sentinel."""
peek(p::DVSParser)::UInt8 = p.position <= length(p.text) ? p.text[p.position] : 0xff
"""Consume Python whitespace, comments and explicit line continuations."""
function whitespace!(p::DVSParser)
    while p.position <= length(p.text)
        token = peek(p)
        if token in (0x20,0x09,0x0c,0x0d,0x0a)
            p.position += 1
        elseif token == UInt8('#')
            while p.position <= length(p.text) && peek(p) != 0x0a
                p.position += 1
            end
        elseif token == UInt8('\\') && p.position < length(p.text) && p.text[p.position+1] == 0x0a
            p.position += 2
        else
            break
        end
    end
    return nothing
end
"""Consume one punctuation token after inert whitespace."""
function consume!(p::DVSParser, token::Char)::Bool
    whitespace!(p)
    if peek(p) == UInt8(token)
        p.position += 1
        return true
    end
    return false
end
"""Parse one scalar or container, bounding structural depth like Python's parser."""
function literal!(p::DVSParser)::DVSLiteral
    whitespace!(p)
    p.position <= length(p.text) || throw(ArgumentError("missing DVS literal"))
    string_start(p) && return DVSString(string_literal!(p))
    token = peek(p)
    if token in (UInt8('('),UInt8('{'))
        p.depth += 1
        p.depth <= 200 || throw(ArgumentError("DVS header nesting exceeds 200"))
        try
            return token == UInt8('{') ? dictionary!(p) : tuple!(p)
        finally
            p.depth -= 1
        end
    end
    signed = token in (UInt8('+'),UInt8('-'))
    if signed
        p.position += 1
        whitespace!(p)
        if peek(p) == UInt8('(')
            operand = literal!(p)
            operand isa DVSNumber && !operand.signed || throw(ArgumentError("invalid DVS unary operand"))
            return DVSNumber(token == UInt8('-') ? -operand.value : operand.value,true)
        end
    end
    start = p.position
    while p.position <= length(p.text) && (UInt8('0') <= peek(p) <= UInt8('9') || UInt8('A') <= peek(p) <= UInt8('Z') || UInt8('a') <= peek(p) <= UInt8('z') || peek(p) == UInt8('_'))
        p.position += 1
    end
    word = String(p.text[start:p.position-1])
    !signed && word == "True" && return DVSBoolean(true)
    !signed && word == "False" && return DVSBoolean(false)
    occursin(r"^(?:0[xX](?:_?[0-9a-fA-F])+|0[oO](?:_?[0-7])+|0[bB](?:_?[01])+|0(?:_?0)*|[1-9](?:_?[0-9])*)$",word) || throw(ArgumentError("invalid DVS integer"))
    cleaned = replace(word,"_"=>"")
    prefixed = occursin(r"^0[xXoObB]",cleaned)
    prefixed || length(cleaned) <= 4300 || throw(ArgumentError("DVS decimal exceeds Python digit limit"))
    number = if prefixed
        base = lowercase(cleaned[2]) == 'x' ? 16 : lowercase(cleaned[2]) == 'o' ? 8 : 2
        parse(BigInt,cleaned[3:end];base=base)
    else
        parse(BigInt,cleaned; base=10)
    end
    return DVSNumber(signed && token == UInt8('-') ? -number : number,signed)
end
"""Distinguish grouped values from actual tuples, including trailing commas."""
function tuple!(p::DVSParser)::DVSLiteral
    p.position += 1
    consume!(p,')') && return DVSTuple(DVSLiteral[])
    first = literal!(p)
    consume!(p,')') && return first
    consume!(p,',') || throw(ArgumentError("missing DVS tuple comma"))
    values = DVSLiteral[first]
    while !consume!(p,')')
        push!(values,literal!(p))
        consume!(p,')') && break
        consume!(p,',') || throw(ArgumentError("missing DVS tuple separator"))
    end
    return DVSTuple(values)
end
"""Refuse duplicate or nonstring dictionary keys before metadata interpretation."""
function dictionary!(p::DVSParser)::DVSDictionary
    p.position += 1
    values = Dict{String,DVSLiteral}()
    consume!(p,'}') && return DVSDictionary(values)
    while true
        key = literal!(p)
        key isa DVSString || throw(ArgumentError("nonstring DVS header key"))
        haskey(values,key.value) && throw(ArgumentError("duplicate DVS header key"))
        consume!(p,':') || throw(ArgumentError("missing DVS dictionary colon"))
        values[key.value] = literal!(p)
        consume!(p,'}') && break
        consume!(p,',') || throw(ArgumentError("missing DVS dictionary separator"))
        consume!(p,'}') && break
    end
    return DVSDictionary(values)
end
