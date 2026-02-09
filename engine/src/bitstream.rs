#[derive(Clone, Debug)]
pub struct BitStreamTensor {
    pub data: Vec<u64>,
    pub length: usize,
}

impl BitStreamTensor {
    pub fn from_words(data: Vec<u64>, length: usize) -> Self {
        Self { data, length }
    }
}

pub fn pack(bits: &[u8]) -> BitStreamTensor {
    let length = bits.len();
    let words = (length + 63) / 64;
    let mut data = vec![0_u64; words];

    for (idx, bit) in bits.iter().copied().enumerate() {
        if bit != 0 {
            data[idx / 64] |= 1_u64 << (idx % 64);
        }
    }

    BitStreamTensor { data, length }
}

pub fn unpack(tensor: &BitStreamTensor) -> Vec<u8> {
    let mut bits = vec![0_u8; tensor.length];

    for idx in 0..tensor.length {
        let word = tensor.data[idx / 64];
        bits[idx] = ((word >> (idx % 64)) & 1) as u8;
    }

    bits
}

pub fn bitwise_and(a: &BitStreamTensor, b: &BitStreamTensor) -> BitStreamTensor {
    assert_eq!(
        a.length, b.length,
        "Bitstream lengths must match for bitwise AND."
    );
    assert_eq!(
        a.data.len(),
        b.data.len(),
        "Packed bitstream shapes must match for bitwise AND."
    );

    let data = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(lhs, rhs)| lhs & rhs)
        .collect();

    BitStreamTensor {
        data,
        length: a.length,
    }
}

pub fn swar_popcount_word(mut x: u64) -> u64 {
    x = x.wrapping_sub((x >> 1) & 0x5555_5555_5555_5555);
    x = (x & 0x3333_3333_3333_3333) + ((x >> 2) & 0x3333_3333_3333_3333);
    x = (x + (x >> 4)) & 0x0f0f_0f0f_0f0f_0f0f;
    (x.wrapping_mul(0x0101_0101_0101_0101) >> 56) as u64
}

pub fn popcount_words_portable(data: &[u64]) -> u64 {
    data.iter().copied().map(swar_popcount_word).sum()
}

pub fn popcount(tensor: &BitStreamTensor) -> u64 {
    popcount_words_portable(&tensor.data)
}

#[cfg(test)]
mod tests {
    use super::{bitwise_and, pack, popcount, unpack};

    #[test]
    fn pack_unpack_roundtrip() {
        let bits = vec![1, 0, 1, 1, 0, 1, 0, 0, 1];
        let packed = pack(&bits);
        let unpacked = unpack(&packed);
        assert_eq!(bits, unpacked);
    }

    #[test]
    fn and_and_popcount() {
        let a = pack(&[1, 0, 1, 1, 0, 0, 1, 1]);
        let b = pack(&[1, 1, 1, 0, 0, 1, 1, 0]);
        let c = bitwise_and(&a, &b);
        assert_eq!(unpack(&c), vec![1, 0, 1, 0, 0, 0, 1, 0]);
        assert_eq!(popcount(&c), 3);
    }
}
