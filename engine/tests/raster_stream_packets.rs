// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Raster frame and whole-container wire evidence

use sc_neurocore_engine::raster_stream::{
    decode_frame, decode_raster, encode_frame, encode_raster,
};

const RASTER: [i8; 9] = [1, 0, 0, 0, 0, 1, 1, 0, 0];
const FRAME: &[u8] = &[0, 3, 0, 3, 2, 5, 2];
const PACKET: &[u8] = b"SSCF\x00\x02\x00\x00\x00\x03\x00\x03\x00\x00\x00\x02\x00\x03\x00\x02\x02\x01\x02\x00\x03\x00\x02\x06\x01";

#[test]
fn matches_independent_channel_masks_and_the_padded_container() {
    assert_eq!(encode_frame(&RASTER, 3, 3).unwrap(), FRAME);
    let decoded = decode_frame(FRAME).unwrap();
    assert_eq!((decoded.rows, decoded.channels), (3, 3));
    assert_eq!(decoded.samples, RASTER);

    let encoded = encode_raster(&RASTER, 3, 3, 2).unwrap();
    assert_eq!(encoded.packet, PACKET);
    assert_eq!(encoded.spikes, 3);
    assert_eq!(encoded.active_channels, 3);
    assert_eq!(encoded.max_frame_bytes, 7);
    assert_eq!(decode_raster(PACKET, 0, 0).unwrap().samples, RASTER);
    assert_eq!(decode_raster(PACKET, 2, 3).unwrap().samples, RASTER[..6]);
}

#[test]
fn refuses_truncation_padding_and_mismatched_dimensions() {
    for prefix in 0..FRAME.len() {
        assert!(decode_frame(&FRAME[..prefix]).is_err());
    }
    for prefix in 0..PACKET.len() {
        assert!(decode_raster(&PACKET[..prefix], 0, 0).is_err());
    }
    let mut extended = PACKET.to_vec();
    extended.push(0);
    assert!(decode_raster(&extended, 0, 0).is_err());
    assert!(decode_raster(PACKET, 3, 2).is_err());
    assert!(decode_raster(PACKET, 4, 3).is_err());
    let mut tampered = PACKET.to_vec();
    *tampered.last_mut().unwrap() |= 2;
    assert!(decode_raster(&tampered, 0, 0).is_err());
    assert!(encode_frame(&[2], 1, 1).is_err());
}
