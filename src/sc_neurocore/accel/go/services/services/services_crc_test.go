// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service packet integrity tests

package main

import (
	"encoding/binary"
	"hash/crc32"
	"testing"
)

func TestEncodeSpikePacketStoresIEEECRC32(t *testing.T) {
	packet := SpikePacket{
		Timestamp: 0x0102030405060708,
		NeuronID:  0x11223344,
		LayerID:   0x5566,
		Polarity:  1,
		Payload:   [9]byte{0, 1, 2, 3, 4, 5, 6, 7, 8},
	}

	frame := EncodeSpikePacket(packet)
	if len(frame) != 28 {
		t.Fatalf("encoded frame length = %d, want 28", len(frame))
	}

	got := binary.BigEndian.Uint32(frame[24:28])
	want := crc32.ChecksumIEEE(frame[:24])
	if got != want {
		t.Fatalf("encoded CRC32 = %#08x, want IEEE CRC32 %#08x", got, want)
	}
}

func TestDecodeSpikePacketCheckedRejectsCorruptCRC(t *testing.T) {
	packet := SpikePacket{
		Timestamp: 123,
		NeuronID:  456,
		LayerID:   7,
		Polarity:  1,
		Payload:   [9]byte{9, 8, 7, 6, 5, 4, 3, 2, 1},
	}

	frame := EncodeSpikePacket(packet)
	frame[3] ^= 0x40

	if _, err := DecodeSpikePacketChecked(frame); err == nil {
		t.Fatal("DecodeSpikePacketChecked accepted a corrupt CRC32 frame")
	}
}

func TestDecodeSpikePacketCheckedRejectsWrongLength(t *testing.T) {
	if _, err := DecodeSpikePacketChecked(make([]byte, 27)); err == nil {
		t.Fatal("DecodeSpikePacketChecked accepted a short frame")
	}
}
