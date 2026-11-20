// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Legacy address-event packet evidence

use sc_neurocore_engine::address_events::{decode_raster, encode_raster};

const PACKET: &[u8] = b"AERX\x00\x00\x00\x14\x00\x00\x00\x03\x00\x00\x00\x03\x01\x00\x00\x00\x00\x00\x02\x00\x07\xff\x00\x07\xff\x00\x04\x01";

#[test]
fn preserves_simultaneous_events_and_timestamp_escapes() {
    let mut raster = vec![0; 60];
    raster[0] = 1;
    raster[2] = 1;
    raster[18 * 3 + 1] = 1;
    let result = encode_raster(&raster, 20, 3, 3, 0).unwrap();
    assert_eq!(result.packet, PACKET);
    assert_eq!((result.events, result.spikes), (3, 3));
    assert_eq!(decode_raster(PACKET, 0, 0).unwrap().samples, raster);
    assert_eq!(decode_raster(PACKET, 1, 2).unwrap().samples, &[1, 0]);
}

#[test]
fn validates_entire_packet_and_reserves_the_escape_address() {
    for cut in 0..PACKET.len() {
        assert!(decode_raster(&PACKET[..cut], 0, 0).is_err());
    }
    let mut extra = PACKET.to_vec();
    extra.extend_from_slice(&[0, 0, 1]);
    assert!(decode_raster(&extra, 1, 1).is_err());

    let mut raster = vec![1; 2 * 256];
    raster[255] = 0;
    let packet = encode_raster(&raster, 2, 256, 16, 8).unwrap().packet;
    assert_eq!(&packet[..4], b"AERI");
    assert_eq!(packet[16], 2);
    assert_eq!(&packet[17..], &[0, 0, 0, 255]);
    assert_eq!(decode_raster(&packet, 0, 0).unwrap().samples, raster);
}
