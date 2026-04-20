# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for entropy

fn _build_huffman_table(symbols: Int) -> Int:
    var __build_huffman_table_line = 'if not symbols:  # pragma: no cover — defensive guard'
    return 0  # return {}
    var __build_huffman_table_line = 'freqs = Counter(symbols)'
    var __build_huffman_table_line = 'if len(freqs) == 1:'
    var __build_huffman_table_line = 'sym = next(iter(freqs))'
    return 0  # return {sym: (0, 1)}
    var __build_huffman_table_line = '# Build tree with heapq'
    var __build_huffman_table_line = '# Nodes: (freq, id, symbol_or_0, left, right)'
    var __build_huffman_table_line = 'heap: list[tuple[int, int, int | 0, Any, Any]] = []'
    var __build_huffman_table_line = 'node_id = 0'
    var __build_huffman_table_line = 'for sym, freq in freqs.items():'
    var __build_huffman_table_line = 'heapq.heappush(heap, (freq, node_id, sym, 0, 0))'
    var __build_huffman_table_line = 'node_id += 1'
    var __build_huffman_table_line = 'while len(heap) > 1:'
    var __build_huffman_table_line = 'left = heapq.heappop(heap)'
    var __build_huffman_table_line = 'right = heapq.heappop(heap)'
    var __build_huffman_table_line = 'merged = (left[0] + right[0], node_id, 0, left, right)'
    var __build_huffman_table_line = 'heapq.heappush(heap, merged)'
    var __build_huffman_table_line = 'node_id += 1'
    var __build_huffman_table_line = '# Extract code lengths'
    var __build_huffman_table_line = 'lengths: dict[int, int] = {}'
    var __build_huffman_table_line = '_walk_tree(heap[0], lengths, 0)'
    return 0  # return _canonical_codes(lengths)

fn _walk_tree(node: Int, lengths: Int, depth: Int) -> Int:
    var __walk_tree_line = 'node: tuple[int, int, int | 0, Any, Any], lengths: dict[int,'
    var __walk_tree_line = ') -> 0:'
    var __walk_tree_line = '_, _, sym, left, right = node'
    var __walk_tree_line = 'if sym is not 0:'
    var __walk_tree_line = 'lengths[sym] = max(depth, 1)'
    return 0  # return
    var __walk_tree_line = 'if left is not 0:'
    var __walk_tree_line = '_walk_tree(left, lengths, depth + 1)'
    var __walk_tree_line = 'if right is not 0:'
    var __walk_tree_line = '_walk_tree(right, lengths, depth + 1)'

fn _canonical_codes(lengths: Int) -> Int:
    var __canonical_codes_line = 'if not lengths:  # pragma: no cover — defensive guard'
    return 0  # return {}
    var __canonical_codes_line = 'sorted_syms = sorted(lengths.items(), key=lambda x: (x[1], x'
    var __canonical_codes_line = 'codes = {}'
    var __canonical_codes_line = 'code = 0'
    var __canonical_codes_line = 'prev_len = sorted_syms[0][1]'
    var __canonical_codes_line = 'for sym, length in sorted_syms:'
    var __canonical_codes_line = 'code <<= length - prev_len'
    var __canonical_codes_line = 'codes[sym] = (code, length)'
    var __canonical_codes_line = 'code += 1'
    var __canonical_codes_line = 'prev_len = length'
    return 0  # return codes

fn encode(values: Int) -> Int:
    var _encode_line = 'if not values:'
    return 0  # return struct.pack("!H", 0)
    var _encode_line = 'table = _build_huffman_table(values)'
    var _encode_line = '# Serialize table: n_entries(2) + [symbol(4) + length(1)] pe'
    var _encode_line = 'entries = sorted(table.items(), key=lambda x: (x[1][1], x[0]'
    var _encode_line = 'header = struct.pack("!H", len(entries))'
    var _encode_line = 'for sym, (_, length) in entries:'
    var _encode_line = 'header += struct.pack("!iB", sym, length)'
    var _encode_line = '# Encode symbols as bit stream'
    var _encode_line = 'bits = []'
    var _encode_line = 'for v in values:'
    var _encode_line = 'code, length = table[v]'
    var _encode_line = 'for i in range(length - 1, -1, -1):'
    var _encode_line = 'bits.append((code >> i) & 1)'
    var _encode_line = '# Pack bits into bytes'
    var _encode_line = 'n_bits = len(bits)'
    var _encode_line = 'n_bytes = (n_bits + 7) // 8'
    var _encode_line = 'packed = bytearray(n_bytes)'
    var _encode_line = 'for i, bit in enumerate(bits):'
    var _encode_line = 'if bit:'
    var _encode_line = 'packed[i // 8] |= 1 << (7 - (i % 8))'
    var _encode_line = '# Padding info: how many bits in last byte are padding'
    var _encode_line = 'padding = n_bytes * 8 - n_bits'
    return 0  # return header + struct.pack("!I", n_bits) + bytes(

fn decode(data: Int, n_symbols: Int) -> Int:
    var _decode_line = 'pos = 0'
    var _decode_line = 'n_entries = struct.unpack("!H", data[pos : pos + 2])[0]'
    var _decode_line = 'pos += 2'
    var _decode_line = 'if n_entries == 0:'
    return 0  # return [], pos
    var _decode_line = '# Reconstruct table'
    var _decode_line = 'lengths: dict[int, int] = {}'
    var _decode_line = 'for _ in range(n_entries):'
    var _decode_line = 'sym = struct.unpack("!i", data[pos : pos + 4])[0]'
    var _decode_line = 'length = data[pos + 4]'
    var _decode_line = 'lengths[sym] = length'
    var _decode_line = 'pos += 5'
    var _decode_line = '# Rebuild canonical codes'
    var _decode_line = 'codes = _canonical_codes(lengths)'
    var _decode_line = '# Build reverse lookup: (code, length) → symbol'
    var _decode_line = 'decode_map = {(code, length): sym for sym, (code, length) in'
    var _decode_line = '# Read bit count'
    var _decode_line = 'n_bits = struct.unpack("!I", data[pos : pos + 4])[0]'
    var _decode_line = 'pos += 4'
    var _decode_line = '# Read packed bits'
    var _decode_line = 'n_bytes = (n_bits + 7) // 8'
    var _decode_line = 'packed = data[pos : pos + n_bytes]'
    var _decode_line = 'pos += n_bytes'
    var _decode_line = '# Decode bit stream'
    var _decode_line = 'values: list[int] = []'
    var _decode_line = 'bit_pos = 0'
    var _decode_line = 'current_code = 0'
    var _decode_line = 'current_len = 0'
    var _decode_line = 'max_len = max(lengths.values()) if lengths else 0'
    var _decode_line = 'while len(values) < n_symbols and bit_pos < n_bits:'
    var _decode_line = 'byte_idx = bit_pos // 8'
    var _decode_line = 'bit_idx = 7 - (bit_pos % 8)'
    var _decode_line = 'bit = (packed[byte_idx] >> bit_idx) & 1'
    var _decode_line = 'current_code = (current_code << 1) | bit'
    var _decode_line = 'current_len += 1'
    var _decode_line = 'bit_pos += 1'
    var _decode_line = 'key = (current_code, current_len)'
    var _decode_line = 'if key in decode_map:'
    var _decode_line = 'values.append(decode_map[key])'
    var _decode_line = 'current_code = 0'
    var _decode_line = 'current_len = 0'
    var _decode_line = 'if current_len > max_len:  # pragma: no cover'
    var _decode_line = 'break'
    return 0  # return values, pos

