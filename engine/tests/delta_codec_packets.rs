// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Delta packet contract tests

use sc_neurocore_engine::codecs::delta::{decode_raster, encode_raster};
use sc_neurocore_engine::codecs::isi::Entropy;

#[test]
fn legacy_layout_retains_first_reference_and_exact_intervals() {
    let raster = [1, 1, 0, 1, 1, 0, 0, 0];
    let packet = [
        b'D', b'S', b'C', b'X', 0, 2, 0, 1, 0, 0, 0, 0, 4, 0, 0, 0, 2, 2, 0, 2, 2, 1, 1,
    ];
    let result = encode_raster(&raster, 4, 2, 2, 1, Entropy::Varint).unwrap();
    assert_eq!(result.packet, packet);
    assert_eq!((result.spike_count, result.residual_count), (4, 2));
    assert_eq!(decode_raster(&packet, 4, 2, 1).unwrap(), raster);
}

#[test]
fn geometry_is_refused_before_the_isi_body_is_read() {
    let packet = [b'D', b'S', b'C', b'X', 0, 2, 0, 2, 0, 1];
    assert_eq!(
        decode_raster(&packet, 4, 3, 1),
        Err("DSCX reference index is outside its group")
    );
    assert_eq!(
        decode_raster(&packet, 4, 2, 1),
        Err("DSCX group count does not match the channel count")
    );
    assert_eq!(
        encode_raster(&[], 0, 65536, 1, 1, Entropy::Varint).err(),
        Some("DSCX group count exceeds uint16")
    );
}
