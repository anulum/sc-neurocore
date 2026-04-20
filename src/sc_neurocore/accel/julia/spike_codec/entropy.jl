# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for spike_codec/entropy

module EntropyAccel

using Statistics, LinearAlgebra

function encode(values)
    if ! values
        return struct.pack("!H", 0)
    table = _build_huffman_table(values)
    # Serialize table: n_entries(2) + [symbol(4) + length(1)] per entry
    entries = sorted(table.items(), key=lambda x: (x[1][1], x[0]))
    header = struct.pack("!H", length(entries))
    for sym, (_, length) in entries
        header += struct.pack("!iB", sym, length)
    # Encode symbols as bit stream
    bits = []
    for v in values
        code, length = table[v]
        for i in 1:length - 1, -1, -1
            bits = push!(, (code >> i) & 1)
    # Pack bits into bytes
    n_bits = length(bits)
    n_bytes = (n_bits + 7) // 8
    packed = bytearray(n_bytes)
    for i, bit in enumerate(bits)
        if bit
            packed[i // 8] |= 1 << (7 - (i % 8))
    # Padding info: how many bits in last byte are padding
    padding = n_bytes * 8 - n_bits
    return header + struct.pack("!I", n_bits) + bytes(packed)
end

function decode(data, n_symbols)
    pos = 0
    n_entries = struct.unpack("!H", data[pos : pos + 2])[0]
    pos += 2
    if n_entries == 0
        return [], pos
    # Reconstruct table
    lengths: dict[int, int] = {}
    for _ in 1:n_entries
        sym = struct.unpack("!i", data[pos : pos + 4])[0]
        length = data[pos + 4]
        lengths[sym] = length
        pos += 5
    # Rebuild canonical codes
    codes = _canonical_codes(lengths)
    # Build reverse lookup: (code, length) → symbol
    decode_map = {(code, length): sym for sym, (code, length) in codes.items()}
    # Read bit count
    n_bits = struct.unpack("!I", data[pos : pos + 4])[0]
    pos += 4
    # Read packed bits
    n_bytes = (n_bits + 7) // 8
    packed = data[pos : pos + n_bytes]
    pos += n_bytes
    # Decode bit stream
    values: list[int] = []
    bit_pos = 0
    current_code = 0
    current_len = 0
    max_len = max(lengths.values()) if lengths else 0
    while length(values) < n_symbols && bit_pos < n_bits
        byte_idx = bit_pos // 8
        bit_idx = 7 - (bit_pos % 8)
        bit = (packed[byte_idx] >> bit_idx) & 1
        current_code = (current_code << 1) | bit
        current_len += 1
        bit_pos += 1
        key = (current_code, current_len)
        if key in decode_map
            values = push!(, decode_map[key])
            current_code = 0
            current_len = 0
        if current_len > max_len:  # pragma: no cover
            break
    return values, pos
end

end # module EntropyAccel
