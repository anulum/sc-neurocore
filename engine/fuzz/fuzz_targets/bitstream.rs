// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fuzz the bitstream pack/unpack primitives.
//!
//! `bitstream::pack` and `bitstream::pack_fast` are two independent implementations of the
//! same operation (bit-vector -> packed `u64` words), so on any input they must produce
//! identical output — a differential check that a fuzzer with arbitrary bytes exercises far
//! beyond the 0/1-only property test. The target also checks that `unpack` recovers each
//! input byte as its `bit != 0` value (the pack/unpack round trip). Any divergence, panic,
//! or index-out-of-bounds aborts under libFuzzer. Build without Z3:
//! `cargo +nightly fuzz build bitstream`.
#![no_main]

use libfuzzer_sys::fuzz_target;
use sc_neurocore_engine::bitstream::{pack, pack_fast, unpack};

fuzz_target!(|data: &[u8]| {
    let packed = pack(data);
    let packed_fast = pack_fast(data);
    // Two independent pack implementations must agree bit-for-bit.
    assert_eq!(
        packed.data, packed_fast.data,
        "pack and pack_fast produced different words"
    );
    assert_eq!(packed.length, packed_fast.length);

    // Unpack recovers each input byte as its 0/1 bit.
    let unpacked = unpack(&packed);
    assert_eq!(unpacked.len(), data.len(), "unpack length mismatch");
    for (i, &byte) in data.iter().enumerate() {
        assert_eq!(
            unpacked[i],
            u8::from(byte != 0),
            "round-trip changed bit {i}"
        );
    }
});
