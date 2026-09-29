# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia converted DVS recording decoder

"""Recognize plain, raw and Unicode Python string prefixes without evaluating them."""
function string_start(p::DVSParser)::Bool
    peek(p) in (0x27,0x22) && return true
    return peek(p) in (UInt8('r'),UInt8('R'),UInt8('u'),UInt8('U')) && p.position < length(p.text) && p.text[p.position+1] in (0x27,0x22)
end
"""Decode and concatenate escaped string literals, retaining UTF8 byte boundaries."""
function string_literal!(p::DVSParser)::String
    out = IOBuffer()
    while true
        whitespace!(p)
        string_start(p) || break
        raw = peek(p) in (UInt8('r'),UInt8('R'))
        if peek(p) in (UInt8('r'),UInt8('R'),UInt8('u'),UInt8('U'))
            p.position += 1
        end
        delimiter = peek(p)
        width = p.position+2 <= length(p.text) && p.text[p.position:p.position+2] == fill(delimiter,3) ? 3 : 1
        p.position += width
        while true
            p.position <= length(p.text) || throw(ArgumentError("unclosed DVS string"))
            if p.position+width-1 <= length(p.text) && all(==(delimiter),p.text[p.position:p.position+width-1])
                p.position += width
                break
            end
            ch = peek(p); p.position += 1
            ch == 0x0a && width == 1 && throw(ArgumentError("newline in DVS string"))
            if ch != UInt8('\\')
                write(out,ch)
                continue
            end
            p.position <= length(p.text) || throw(ArgumentError("unterminated DVS escape"))
            escaped = peek(p);p.position += 1
            if raw
                write(out,UInt8('\\'),escaped)
            else
                string_escape!(p,out,escaped)
            end
        end
    end
    text = String(take!(out))
    isvalid(text) || throw(ArgumentError("invalid DVS Unicode string"))
    return text
end
"""Decode basic, octal, hexadecimal and relevant named Unicode escapes."""
function string_escape!(p::DVSParser,out::IOBuffer,ch::UInt8)
    if ch == 0x0a
        return nothing
    elseif ch in (UInt8('\\'),0x27,0x22)
        write(out,ch)
    elseif ch in codeunits("abfnrtv")
        index = findfirst(==(ch),codeunits("abfnrtv"))
        write(out,UInt8((7,8,12,10,13,9,11)[index]))
    elseif ch in codeunits("xuU")
        width = ch == UInt8('x') ? 2 : ch == UInt8('u') ? 4 : 8
        p.position+width-1 <= length(p.text) || throw(ArgumentError("truncated DVS hex escape"))
        digits = String(p.text[p.position:p.position+width-1])
        occursin(r"^[0-9a-fA-F]+$",digits) || throw(ArgumentError("invalid DVS hex escape"))
        number = parse(UInt32,digits;base=16)
        number <= 0x10ffff && !(0xd800 <= number <= 0xdfff) || throw(ArgumentError("invalid DVS Unicode scalar"))
        print(out,Char(number));p.position += width
    elseif ch == UInt8('N')
        peek(p) == UInt8('{') || throw(ArgumentError("invalid DVS named escape"))
        p.position += 1
        start = p.position
        while p.position <= length(p.text) && peek(p) != UInt8('}')
            p.position += 1
        end
        p.position <= length(p.text) || throw(ArgumentError("unclosed DVS named escape"))
        name = String(p.text[start:p.position-1]);p.position += 1
        print(out,named_ascii(name))
    elseif UInt8('0') <= ch <= UInt8('7')
        number = Int(ch-UInt8('0'))
        for _ in 1:2
            UInt8('0') <= peek(p) <= UInt8('7') || break
            number = number*8+Int(peek(p)-UInt8('0'));p.position += 1
        end
        print(out,Char(number))
    else
        write(out,UInt8('\\'),ch)
    end
    return nothing
end
"""Resolve Unicode names capable of forming admitted ASCII keys and descriptors."""
function named_ascii(source::String)::Char
    name = uppercase(source)
    for (word,value) in (("LESS-THAN SIGN",'<'),("GREATER-THAN SIGN",'>'),("EQUALS SIGN",'='),("VERTICAL LINE",'|'),("PLUS SIGN",'+'),("HYPHEN-MINUS",'-'),("QUESTION MARK",'?'),("LOW LINE",'_'))
        name == word && return value
    end
    for (prefix,lower) in (("LATIN SMALL LETTER ",true),("LATIN CAPITAL LETTER ",false))
        if startswith(name,prefix)
            suffix = name[length(prefix)+1:end]
            length(suffix) == 1 && 'A' <= only(suffix) <= 'Z' || throw(ArgumentError("invalid DVS letter name"))
            return lower ? lowercase(only(suffix)) : only(suffix)
        end
    end
    for (index,word) in enumerate(("ZERO","ONE","TWO","THREE","FOUR","FIVE","SIX","SEVEN","EIGHT","NINE"))
        name == "DIGIT "*word && return Char(Int('0')+index-1)
    end
    throw(ArgumentError("unsupported DVS Unicode name"))
end
