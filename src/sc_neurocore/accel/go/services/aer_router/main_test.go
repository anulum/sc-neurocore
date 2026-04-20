// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — AER Interconnect Tests

package main

import (
	"testing"
)

func TestRouterRegistration(t *testing.T) {
	r := NewRouter("0") // OS-assigned port
	defer r.Close()

	if err := r.RegisterRoute(100, "127.0.0.1:9001"); err != nil {
		t.Fatal(err)
	}

	if r.RouteCount() != 1 {
		t.Errorf("expected 1 route, got %d", r.RouteCount())
	}
}

func TestRouterUnregister(t *testing.T) {
	r := NewRouter("0")
	defer r.Close()

	r.RegisterRoute(100, "127.0.0.1:9001")
	r.UnregisterRoute(100)

	if r.RouteCount() != 0 {
		t.Errorf("expected 0 routes after unregister, got %d", r.RouteCount())
	}
}

func TestDispatchToUnregisteredTarget(t *testing.T) {
	r := NewRouter("0")
	defer r.Close()

	p := SpikePacket{SourceID: 1, TargetID: 999, Sequence: 1}
	if r.DispatchSpike(p) {
		t.Error("dispatch to unregistered target should return false")
	}
}

func TestDispatchIncrementsStats(t *testing.T) {
	r := NewRouter("0")
	defer r.Close()

	r.RegisterRoute(100, "127.0.0.1:9001")

	p := SpikePacket{SourceID: 1, TargetID: 100, Sequence: 1}
	ok := r.DispatchSpike(p)
	if !ok {
		t.Fatal("dispatch should succeed for registered target")
	}

	if r.TotalSent() != 1 {
		t.Errorf("expected 1 sent, got %d", r.TotalSent())
	}
}

func TestAckClears(t *testing.T) {
	r := NewRouter("0")
	defer r.Close()

	r.RegisterRoute(100, "127.0.0.1:9001")

	p := SpikePacket{SourceID: 1, TargetID: 100, Sequence: 42}
	r.DispatchSpike(p)

	if r.PendingCount() != 1 {
		t.Errorf("expected 1 pending, got %d", r.PendingCount())
	}

	r.AckReceived(42)

	if r.PendingCount() != 0 {
		t.Errorf("expected 0 pending after ACK, got %d", r.PendingCount())
	}
	if r.TotalAcked() != 1 {
		t.Errorf("expected 1 acked, got %d", r.TotalAcked())
	}
}

func TestPacketEncodeDecode(t *testing.T) {
	orig := SpikePacket{
		SourceID:  1,
		TargetID:  100,
		Timestamp: 1234567890,
		SpikeLen:  256,
		Sequence:  42,
	}

	data, err := EncodePacket(orig)
	if err != nil {
		t.Fatal(err)
	}

	if len(data) != PacketSize {
		t.Fatalf("expected %d bytes, got %d", PacketSize, len(data))
	}

	decoded, err := DecodePacket(data)
	if err != nil {
		t.Fatal(err)
	}

	if decoded != orig {
		t.Errorf("roundtrip mismatch: got %+v, want %+v", decoded, orig)
	}
}

func TestMultiRouteDispatch(t *testing.T) {
	r := NewRouter("0")
	defer r.Close()

	r.RegisterRoute(10, "127.0.0.1:9001")
	r.RegisterRoute(20, "127.0.0.1:9002")
	r.RegisterRoute(30, "127.0.0.1:9003")

	if r.RouteCount() != 3 {
		t.Fatalf("expected 3 routes, got %d", r.RouteCount())
	}

	for i := uint32(0); i < 3; i++ {
		targets := []uint32{10, 20, 30}
		p := SpikePacket{SourceID: 1, TargetID: targets[i], Sequence: uint64(i)}
		if !r.DispatchSpike(p) {
			t.Errorf("dispatch to target %d should succeed", targets[i])
		}
	}

	if r.TotalSent() != 3 {
		t.Errorf("expected 3 sent, got %d", r.TotalSent())
	}
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

func BenchmarkRouterDispatch(b *testing.B) {
	r := NewRouter("0")
	defer r.Close()

	r.RegisterRoute(100, "127.0.0.1:9001")
	p := SpikePacket{SourceID: 1, TargetID: 100, Timestamp: 12345, SpikeLen: 1}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p.Sequence = uint64(i)
		r.DispatchSpike(p)
	}
}

func BenchmarkPacketEncode(b *testing.B) {
	p := SpikePacket{SourceID: 1, TargetID: 100, Timestamp: 12345, SpikeLen: 256, Sequence: 42}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		EncodePacket(p)
	}
}
