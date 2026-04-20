// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — tinySC Hamming(7,4) ECC (no_std)

//! Hamming(7,4) error-correcting code, bit-compatible with `ScDoctor`.

/// Hamming(7,4) encoder.
#[inline]
pub fn encode(data: u8) -> u8 {
    let d1 = (data >> 3) & 1;
    let d2 = (data >> 2) & 1;
    let d3 = (data >> 1) & 1;
    let d4 = data & 1;
    let p1 = d1 ^ d2 ^ d4;
    let p2 = d1 ^ d3 ^ d4;
    let p3 = d2 ^ d3 ^ d4;
    (p1 << 6) | (p2 << 5) | (d1 << 4) | (p3 << 3) | (d2 << 2) | (d3 << 1) | d4
}

/// Hamming(7,4) decoder with single-bit error correction.
#[inline]
pub fn decode(encoded: u8) -> u8 {
    let p1 = (encoded >> 6) & 1;
    let p2 = (encoded >> 5) & 1;
    let d1 = (encoded >> 4) & 1;
    let p3 = (encoded >> 3) & 1;
    let d2 = (encoded >> 2) & 1;
    let d3 = (encoded >> 1) & 1;
    let d4 = encoded & 1;

    let s1 = p1 ^ d1 ^ d2 ^ d4;
    let s2 = p2 ^ d1 ^ d3 ^ d4;
    let s3 = p3 ^ d2 ^ d3 ^ d4;
    let syndrome = (s3 << 2) | (s2 << 1) | s1;

    let corrected = if syndrome > 0 && syndrome <= 7 {
        let positions: [u8; 7] = [6, 5, 4, 3, 2, 1, 0];
        encoded ^ (1 << positions[(syndrome - 1) as usize])
    } else {
        encoded
    };

    let cd1 = (corrected >> 4) & 1;
    let cd2 = (corrected >> 2) & 1;
    let cd3 = (corrected >> 1) & 1;
    let cd4 = corrected & 1;
    (cd1 << 3) | (cd2 << 2) | (cd3 << 1) | cd4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_all_patterns() {
        for data in 0u8..16 {
            assert_eq!(decode(encode(data)), data, "failed for {data}");
        }
    }

    #[test]
    fn single_bit_correction() {
        let data: u8 = 0b1011;
        let enc = encode(data);
        for bit in 0..7u8 {
            let corrupted = enc ^ (1 << bit);
            assert_eq!(decode(corrupted), data, "failed bit {bit}");
        }
    }

    #[test]
    fn encoded_fits_7_bits() {
        for data in 0u8..16 {
            assert!(encode(data) < 128);
        }
    }
}
