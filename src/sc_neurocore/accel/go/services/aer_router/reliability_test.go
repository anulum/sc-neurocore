// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — AER acknowledgement reliability tests

package main

import (
	"bytes"
	"encoding/binary"
	"net"
	"sync/atomic"
	"testing"
	"time"
)

// expectUDP verifies a complete frame received by the actual destination socket.
func expectUDP(t *testing.T, conn *net.UDPConn, want []byte) {
	t.Helper()
	var buffer [64]byte
	n, _, err := conn.ReadFromUDP(buffer[:])
	if err != nil || !bytes.Equal(buffer[:n], want) {
		t.Fatalf("datagram = %x, %v; want %x", buffer[:n], err, want)
	}
}

// TestInflightSequenceKeepsItsRoute checks the origin and accounting of UDP acknowledgements.
func TestInflightSequenceKeepsItsRoute(t *testing.T) {
	router := NewRouter("0")
	t.Cleanup(func() { _ = router.Close() })
	first, second := loopbackUDP(t), loopbackUDP(t)
	if err := router.RegisterRoute(10, first.LocalAddr().String()); err != nil {
		t.Fatal(err)
	}
	if err := router.RegisterRoute(20, second.LocalAddr().String()); err != nil {
		t.Fatal(err)
	}
	go router.Listen()
	addr := &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1), Port: router.conn.LocalAddr().(*net.UDPAddr).Port}
	p := SpikePacket{SourceID: 4, TargetID: 10, Sequence: 5}
	if !router.DispatchSpike(p) {
		t.Fatal("first dispatch failed")
	}
	firstDelivery := readDeliveredSpike(t, first, p)
	p.TargetID = 20
	if router.DispatchSpike(p) {
		t.Fatal("a second destination reused a live sequence")
	}
	ack := binary.BigEndian.AppendUint64(nil, firstDelivery)
	if _, err := second.WriteToUDP(ack, addr); err != nil {
		t.Fatal(err)
	}
	barrierPacket := SpikePacket{TargetID: 20, Sequence: 6}
	barrier, err := EncodePacket(barrierPacket)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := second.WriteToUDP(barrier, addr); err != nil {
		t.Fatal(err)
	}
	secondDelivery := readDeliveredSpike(t, second, barrierPacket)
	expectUDP(t, second, barrier[PacketSize-8:])
	if router.TotalAcked() != 0 || router.PendingCount() != 2 {
		t.Fatal("the wrong endpoint changed pending ACK accounting")
	}
	if _, err := first.WriteToUDP(ack, addr); err != nil {
		t.Fatal(err)
	}
	if _, err := second.WriteToUDP(binary.BigEndian.AppendUint64(nil, secondDelivery), addr); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(time.Second)
	for router.TotalAcked() < 2 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if router.TotalAcked() != 2 || router.PendingCount() != 0 || atomic.LoadUint64(&router.stats[10].Acked) != 1 {
		t.Fatal("the expected endpoint did not clear its own sequences")
	}
	if atomic.LoadUint64(&router.stats[20].Acked) != 1 || !router.DispatchSpike(p) {
		t.Fatal("sequence ownership was not released after the correct ACK")
	}
}

// TestMissingAcknowledgementsExpire verifies rejection at the limit and recovery after expiry.
func TestMissingAcknowledgementsExpire(t *testing.T) {
	router := NewRouter("0")
	t.Cleanup(func() { _ = router.Close() })
	dst := loopbackUDP(t)
	if err := router.RegisterRoute(10, dst.LocalAddr().String()); err != nil {
		t.Fatal(err)
	}
	const limit = 4096
	for sequence := uint64(0); sequence < limit; sequence++ {
		if !router.DispatchSpike(SpikePacket{TargetID: 10, Sequence: sequence}) {
			t.Fatalf("pending sequence %d refused below the limit", sequence)
		}
	}
	if router.DispatchSpike(SpikePacket{TargetID: 10, Sequence: limit}) {
		t.Fatal("a missing-ACK workload exceeded the pending limit")
	}
	time.Sleep(2 * time.Second)
	router.AckReceived(0)
	if router.TotalAcked() != 0 || router.PendingCount() != 0 || router.stats[10].Dropped != limit+1 {
		t.Fatalf("expired dispatches were not accounted for: %+v", router.stats[10])
	}
	if !router.DispatchSpike(SpikePacket{TargetID: 10, Sequence: 0}) {
		t.Fatal("dispatch did not recover after the missing ACKs expired")
	}
}

// TestExpiredSequenceReuseKeepsDeliveryIdentity verifies delayed ACKs across a shared destination.
func TestExpiredSequenceReuseKeepsDeliveryIdentity(t *testing.T) {
	router := NewRouter("0")
	t.Cleanup(func() { _ = router.Close() })
	receiver := loopbackUDP(t)
	for _, target := range []uint32{10, 20} {
		if err := router.RegisterRoute(target, receiver.LocalAddr().String()); err != nil {
			t.Fatal(err)
		}
	}
	go router.Listen()
	address := &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1), Port: router.conn.LocalAddr().(*net.UDPAddr).Port}
	packet := SpikePacket{SourceID: 4, TargetID: 10, Sequence: 25}
	if !router.DispatchSpike(packet) {
		t.Fatal("initial dispatch failed")
	}
	oldToken := readDeliveredSpike(t, receiver, packet)
	time.Sleep(2 * time.Second)
	if router.PendingCount() != 0 || atomic.LoadUint64(&router.stats[10].Dropped) != 1 {
		t.Fatal("initial dispatch did not expire")
	}
	if err := receiver.SetReadDeadline(time.Now().Add(time.Second)); err != nil {
		t.Fatal(err)
	}
	packet.TargetID = 20
	if !router.DispatchSpike(packet) {
		t.Fatal("new dispatch after expiry failed")
	}
	newToken := readDeliveredSpike(t, receiver, packet)
	if newToken == oldToken {
		t.Fatal("the new delivery reused a retired downstream token")
	}
	if _, err := receiver.WriteToUDP(binary.BigEndian.AppendUint64(nil, oldToken), address); err != nil {
		t.Fatal(err)
	}
	barrierPacket := SpikePacket{TargetID: 10, Sequence: 26}
	barrier, err := EncodePacket(barrierPacket)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := receiver.WriteToUDP(barrier, address); err != nil {
		t.Fatal(err)
	}
	barrierToken := readDeliveredSpike(t, receiver, barrierPacket)
	expectUDP(t, receiver, barrier[PacketSize-8:])
	if router.TotalAcked() != 0 || router.PendingCount() != 2 {
		t.Fatal("a delayed ACK changed accounting for a new delivery")
	}
	for _, token := range []uint64{newToken, barrierToken} {
		if _, err := receiver.WriteToUDP(binary.BigEndian.AppendUint64(nil, token), address); err != nil {
			t.Fatal(err)
		}
	}
	deadline := time.Now().Add(time.Second)
	for router.TotalAcked() < 2 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if router.TotalAcked() != 2 || router.PendingCount() != 0 ||
		atomic.LoadUint64(&router.stats[10].Acked) != 1 || atomic.LoadUint64(&router.stats[20].Acked) != 1 {
		t.Fatal("fresh wire acknowledgements were not attributed to their individual routes")
	}
}

// TestDeliverySequenceExhaustion confirms wire tokens are never recycled after the counter reaches its limit.
func TestDeliverySequenceExhaustion(t *testing.T) {
	router := NewRouter("0")
	t.Cleanup(func() { _ = router.Close() })
	receiver := loopbackUDP(t)
	if err := router.RegisterRoute(10, receiver.LocalAddr().String()); err != nil {
		t.Fatal(err)
	}
	router.deliveryBase = maxDeliverySequence - 1
	router.nextDelivery = maxDeliverySequence - 1
	packet := SpikePacket{TargetID: 10, Sequence: 25}
	if !router.DispatchSpike(packet) {
		t.Fatal("last delivery token was refused")
	}
	if got := readDeliveredSpike(t, receiver, packet); got != maxDeliverySequence-1 {
		t.Fatalf("last token = %d", got)
	}
	router.AckReceived(packet.Sequence)
	packet.Sequence++
	if router.DispatchSpike(packet) || router.TotalSent() != 1 || router.TotalAcked() != 1 {
		t.Fatal("the exhausted token sequence admitted a replacement")
	}
}
