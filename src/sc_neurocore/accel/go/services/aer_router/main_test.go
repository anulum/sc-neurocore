// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — AER Interconnect Tests

package main

import (
	"bytes"
	"encoding/binary"
	"math"
	"net"
	"testing"
	"time"
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

	r.AckReceived(999)
	r.AckReceived(42)
	r.AckReceived(42)

	if r.PendingCount() != 0 {
		t.Errorf("expected 0 pending after ACK, got %d", r.PendingCount())
	}
	if r.TotalAcked() != 1 {
		t.Errorf("expected 1 acked, got %d", r.TotalAcked())
	}
	if stats := r.stats[100]; stats.Acked != 1 {
		t.Errorf("expected 1 acknowledged on the target route, got %d", stats.Acked)
	}
}

// loopbackUDP opens a socket owned by a test with an OS-assigned port.
func loopbackUDP(t testing.TB) *net.UDPConn {
	t.Helper()
	conn, err := net.ListenUDP("udp4", &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1)})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	if err := conn.SetReadDeadline(time.Now().Add(time.Second)); err != nil {
		t.Fatal(err)
	}
	return conn
}

// TestListenForwardsRegisteredPacket covers the actual UDP data path.
func TestListenForwardsRegisteredPacket(t *testing.T) {
	router := NewRouter("0")
	t.Cleanup(func() { _ = router.Close() })
	destination := loopbackUDP(t)
	source := loopbackUDP(t)
	if err := router.RegisterRoute(12, destination.LocalAddr().String()); err != nil {
		t.Fatal(err)
	}
	go router.Listen()

	// Python clients can view the top half of the 64-bit sequence field as signed.
	packet := SpikePacket{SourceID: 8, TargetID: 12, Timestamp: 17, SpikeLen: 9, Sequence: math.MaxUint64}
	data, err := EncodePacket(packet)
	if err != nil {
		t.Fatal(err)
	}
	port := router.conn.LocalAddr().(*net.UDPAddr).Port
	addr := &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1), Port: port}
	if _, err := source.WriteToUDP(data, addr); err != nil {
		t.Fatal(err)
	}

	delivery := readDeliveredSpike(t, destination, packet)
	var buffer [64]byte
	n, _, err := source.ReadFromUDP(buffer[:])
	if err != nil || !bytes.Equal(data[PacketSize-8:], buffer[:n]) {
		t.Fatalf("upstream ACK = %x, %v; want sequence echo", buffer[:n], err)
	}
	if _, err := destination.WriteToUDP(binary.BigEndian.AppendUint64(nil, delivery), addr); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(time.Second)
	for router.TotalAcked() == 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if router.TotalAcked() != 1 || router.PendingCount() != 0 {
		t.Fatal("downstream ACK did not clear the forwarded sequence")
	}
}

// TestListenReturnsOnClose ensures router instances own no live read loop after shutdown.
func TestListenReturnsOnClose(t *testing.T) {
	router := NewRouter("0")
	done := make(chan struct{})
	go func() {
		router.Listen()
		close(done)
	}()
	if err := router.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("listening goroutine survived socket close")
	}
}

// TestWriteFailureDoesNotLeavePending ensures failed sends cannot accept later ACKs.
func TestWriteFailureDoesNotLeavePending(t *testing.T) {
	router := NewRouter("0")
	address := loopbackUDP(t).LocalAddr().String()
	if err := router.RegisterRoute(5, address); err != nil {
		t.Fatal(err)
	}
	if err := router.Close(); err != nil {
		t.Fatal(err)
	}
	if router.DispatchSpike(SpikePacket{TargetID: 5, Sequence: 8}) {
		t.Fatal("closed socket reported successful dispatch")
	}
	if router.PendingCount() != 0 {
		t.Fatal("failed write left a pending sequence")
	}
	router.AckReceived(8)
	if router.TotalAcked() != 0 {
		t.Fatal("ACK for a failed write was counted")
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

	destination := loopbackUDP(b)
	if err := destination.SetReadDeadline(time.Now().Add(time.Minute)); err != nil {
		b.Fatal(err)
	}
	if err := r.RegisterRoute(100, destination.LocalAddr().String()); err != nil {
		b.Fatal(err)
	}
	p := SpikePacket{SourceID: 1, TargetID: 100, Timestamp: 12345, SpikeLen: 1}
	var buffer [PacketSize]byte

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p.Sequence = uint64(i)
		if !r.DispatchSpike(p) {
			b.Fatal("dispatch was refused")
		}
		if n, _, err := destination.ReadFromUDP(buffer[:]); err != nil || n != PacketSize {
			b.Fatalf("forwarded packet has %d bytes: %v", n, err)
		}
		r.AckReceived(p.Sequence)
	}
	b.StopTimer()
	if r.TotalSent() != uint64(b.N) || r.TotalAcked() != uint64(b.N) || r.PendingCount() != 0 {
		b.Fatal("dispatch/ACK totals do not match completed iterations")
	}
}

func BenchmarkPacketEncode(b *testing.B) {
	p := SpikePacket{SourceID: 1, TargetID: 100, Timestamp: 12345, SpikeLen: 256, Sequence: 42}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		EncodePacket(p)
	}
}

// readDeliveredSpike checks the preserved event fields and returns the wire token.
func readDeliveredSpike(t *testing.T, destination *net.UDPConn, want SpikePacket) uint64 {
	t.Helper()
	var buffer [64]byte
	n, _, err := destination.ReadFromUDP(buffer[:])
	if err != nil {
		t.Fatal(err)
	}
	if n != PacketSize {
		t.Fatalf("forwarded packet has %d bytes", n)
	}
	packet, err := DecodePacket(buffer[:n])
	if err != nil {
		t.Fatal(err)
	}
	want.Sequence = packet.Sequence
	if packet != want {
		t.Fatalf("forwarded event = %+v; want %+v", packet, want)
	}
	return packet.Sequence
}
