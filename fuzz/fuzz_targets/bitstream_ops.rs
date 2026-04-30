// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

#![no_main]

use libfuzzer_sys::fuzz_target;
use sc_neurocore_engine::bitstream::{
    bitwise_and, pack, pack_fast, popcount, unpack, BitStreamTensor,
};

fuzz_target!(|data: &[u8]| {
    if data.len() > 4096 {
        return;
    }

    let bits: Vec<u8> = data.iter().map(|byte| byte & 1).collect();
    let packed = pack(&bits);
    let packed_fast = pack_fast(&bits);

    assert_eq!(packed.length, bits.len());
    assert_eq!(packed_fast.length, bits.len());
    assert_eq!(packed.data, packed_fast.data);
    assert_eq!(unpack(&packed), bits);
    assert_eq!(
        popcount(&packed),
        bits.iter().map(|bit| u64::from(*bit)).sum::<u64>()
    );

    if bits.is_empty() {
        return;
    }

    let mut rotated = packed.clone();
    rotated.rotate_right(data[0] as usize);
    assert_eq!(rotated.length, packed.length);
    assert_eq!(unpack(&rotated).len(), bits.len());

    let and_self = bitwise_and(&packed, &packed);
    assert_eq!(and_self.data, packed.data);
    assert_eq!(and_self.length, packed.length);

    let tensor = BitStreamTensor::from_words(packed.data.clone(), packed.length);
    assert_eq!(tensor.hamming_distance(&packed), 0.0);
});
