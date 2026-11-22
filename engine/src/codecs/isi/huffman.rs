// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Canonical Huffman ISI body

//! Canonical Huffman coding with the historical Python heap tie order.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use super::framing::PacketReader;

const MAX_CODE_LENGTH: u8 = 63;
type Lengths = Vec<(u8, i32)>;

enum Node {
    Symbol(i32),
    Pair(usize, usize),
}

fn lengths(values: &[i32]) -> Result<Lengths, &'static str> {
    let mut positions: HashMap<i32, usize> = HashMap::new();
    let mut frequencies = Vec::<usize>::new();
    let mut nodes = Vec::new();
    for &symbol in values {
        let position = *positions.entry(symbol).or_insert_with(|| {
            let position = nodes.len();
            nodes.push(Node::Symbol(symbol));
            frequencies.push(0);
            position
        });
        frequencies[position] += 1;
    }
    if u16::try_from(nodes.len()).is_err() {
        return Err("Huffman table exceeds uint16");
    }
    let mut heap: BinaryHeap<_> = frequencies
        .into_iter()
        .enumerate()
        .map(|(index, frequency)| Reverse((frequency, index)))
        .collect();
    while heap.len() > 1 {
        let Reverse((left_freq, left)) = heap.pop().ok_or("missing Huffman node")?;
        let Reverse((right_freq, right)) = heap.pop().ok_or("missing Huffman node")?;
        let index = nodes.len();
        nodes.push(Node::Pair(left, right));
        heap.push(Reverse((left_freq + right_freq, index)));
    }
    let mut result = Vec::with_capacity(positions.len());
    if let Some(Reverse((_, root))) = heap.pop() {
        let mut stack = vec![(root, 0)];
        while let Some((index, depth)) = stack.pop() {
            match nodes[index] {
                Node::Symbol(symbol) => result.push((depth.max(1), symbol)),
                Node::Pair(left, right) => {
                    if depth == MAX_CODE_LENGTH {
                        return Err("Huffman code length exceeds limit");
                    }
                    stack.push((right, depth + 1));
                    stack.push((left, depth + 1));
                }
            }
        }
    }
    result.sort_unstable();
    Ok(result)
}

fn canonical_codes(lengths: &[(u8, i32)]) -> Result<Vec<u64>, &'static str> {
    let mut next = 0u64;
    let mut previous = 0;
    let mut codes = Vec::with_capacity(lengths.len());
    for &(length, _) in lengths {
        if !(1..=MAX_CODE_LENGTH).contains(&length) {
            return Err("invalid Huffman code length");
        }
        let shift = length - previous;
        // Compare before shifting so a malformed table cannot wrap the code.
        if next >= (1u64 << previous) {
            return Err("oversubscribed Huffman table");
        }
        let code = next << shift;
        codes.push(code);
        next = code + 1;
        previous = length;
    }
    Ok(codes)
}

pub(super) fn encode(values: &[i32], output: &mut Vec<u8>) -> Result<(), &'static str> {
    let lengths = lengths(values)?;
    let count = u16::try_from(lengths.len()).map_err(|_| "Huffman table exceeds uint16")?;
    output.extend_from_slice(&count.to_be_bytes());
    if count == 0 {
        return Ok(());
    }
    let codes = canonical_codes(&lengths)?;
    let mut by_symbol = HashMap::with_capacity(lengths.len());
    for ((length, symbol), code) in lengths.into_iter().zip(codes) {
        output.extend_from_slice(&symbol.to_be_bytes());
        output.push(length);
        by_symbol.insert(symbol, (length, code));
    }
    let bit_count = values.iter().try_fold(0u32, |bits, symbol| {
        bits.checked_add(u32::from(by_symbol[symbol].0))
            .ok_or("Huffman bit count must fit uint32")
    })?;
    output.extend_from_slice(&bit_count.to_be_bytes());
    let start = output.len();
    let byte_count = (bit_count as usize).div_ceil(8);
    output.resize(start + byte_count, 0);
    let mut offset = 0;
    for symbol in values {
        let (length, code) = by_symbol[symbol];
        for shift in (0..length).rev() {
            output[start + offset / 8] |= (((code >> shift) & 1) as u8) << (7 - offset % 8);
            offset += 1;
        }
    }
    Ok(())
}

pub(super) fn decode(
    reader: &mut PacketReader<'_>,
    symbol_count: usize,
) -> Result<Vec<i32>, &'static str> {
    let count = usize::from(reader.u16()?);
    if count == 0 && symbol_count == 0 {
        return Ok(Vec::new());
    }
    if count == 0 || count > symbol_count {
        return Err("Huffman table does not match the symbol count");
    }
    let mut header = PacketReader::new(reader.take(count * 5)?);
    let mut lengths = Vec::with_capacity(count);
    let mut unique = std::collections::HashSet::with_capacity(count);
    for _ in 0..count {
        let symbol = header.u32()? as i32;
        if !unique.insert(symbol) {
            return Err("duplicate Huffman symbol");
        }
        lengths.push((header.take(1)?[0], symbol));
    }
    lengths.sort_unstable();
    let codes = canonical_codes(&lengths)?;
    let by_code: HashMap<_, _> = lengths
        .into_iter()
        .zip(codes)
        .map(|((length, symbol), code)| ((length, code), symbol))
        .collect();

    let bits = reader.u32()? as usize;
    if bits < symbol_count {
        return Err("truncated Huffman symbols");
    }
    let packed = reader.take(bits.div_ceil(8))?;
    if !bits.is_multiple_of(8) && packed[packed.len() - 1] & ((1 << (8 - bits % 8)) - 1) != 0 {
        return Err("nonzero Huffman padding");
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(symbol_count)
        .map_err(|_| "Huffman allocation failed")?;
    let mut code = 0;
    let mut length = 0;
    for bit in 0..bits {
        code = (code << 1) | u64::from((packed[bit / 8] >> (7 - bit % 8)) & 1);
        length += 1;
        if let Some(&symbol) = by_code.get(&(length, code)) {
            if values.len() == symbol_count {
                return Err("excess Huffman symbols");
            }
            values.push(symbol);
            code = 0;
            length = 0;
        } else if length == MAX_CODE_LENGTH {
            return Err("invalid Huffman code");
        }
    }
    if values.len() != symbol_count || length != 0 {
        return Err("incomplete Huffman symbol stream");
    }
    Ok(values)
}
