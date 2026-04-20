// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — AER Interconnect Router

// Package main implements an AER-over-UDP multi-FPGA spike router
// with dynamic routing, ACK-based reliability, sequence tracking,
// and per-route latency statistics.
//
// Protocol: Each SpikePacket is serialized as a fixed-size big-endian
// binary frame (28 bytes). ACKs are 8-byte sequence echoes.
package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"log"
	"net"
	"sync"
	"sync/atomic"
	"time"
)

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

// SpikePacket is the wire format for an AER spike event.
type SpikePacket struct {
	SourceID  uint32
	TargetID  uint32
	Timestamp uint64
	SpikeLen  uint32
	Sequence  uint64
}

// PacketSize is the serialized size of SpikePacket in bytes.
const PacketSize = 28

// RouteStats tracks per-route delivery statistics.
type RouteStats struct {
	Dispatched uint64
	Acked      uint64
	Dropped    uint64
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

// Router manages route registration, spike dispatch, and ACK tracking.
type Router struct {
	mu      sync.RWMutex
	routes  map[uint32]*net.UDPAddr
	conn    *net.UDPConn
	pending map[uint64]time.Time
	pmu     sync.Mutex

	stats     map[uint32]*RouteStats
	totalSent atomic.Uint64
	totalAck  atomic.Uint64
}

// NewRouter creates a router listening on the given UDP port.
func NewRouter(port string) *Router {
	addr, err := net.ResolveUDPAddr("udp", ":"+port)
	if err != nil {
		log.Fatalf("[router] resolve addr: %v", err)
	}
	conn, err := net.ListenUDP("udp", addr)
	if err != nil {
		log.Fatalf("[router] listen: %v", err)
	}
	return &Router{
		routes:  make(map[uint32]*net.UDPAddr),
		conn:    conn,
		pending: make(map[uint64]time.Time),
		stats:   make(map[uint32]*RouteStats),
	}
}

// RegisterRoute maps a neuron ID to a destination UDP address.
func (r *Router) RegisterRoute(neuronID uint32, addrStr string) error {
	udpAddr, err := net.ResolveUDPAddr("udp", addrStr)
	if err != nil {
		return fmt.Errorf("resolve %q: %w", addrStr, err)
	}
	r.mu.Lock()
	r.routes[neuronID] = udpAddr
	if _, ok := r.stats[neuronID]; !ok {
		r.stats[neuronID] = &RouteStats{}
	}
	r.mu.Unlock()
	log.Printf("[router] route registered: neuron %d → %s", neuronID, addrStr)
	return nil
}

// UnregisterRoute removes a route for the given neuron ID.
func (r *Router) UnregisterRoute(neuronID uint32) {
	r.mu.Lock()
	delete(r.routes, neuronID)
	r.mu.Unlock()
}

// RouteCount returns the number of registered routes.
func (r *Router) RouteCount() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.routes)
}

// DispatchSpike sends a spike packet to the registered target.
func (r *Router) DispatchSpike(p SpikePacket) bool {
	r.mu.RLock()
	target, ok := r.routes[p.TargetID]
	st := r.stats[p.TargetID]
	r.mu.RUnlock()

	if !ok {
		return false
	}

	buf := new(bytes.Buffer)
	if err := binary.Write(buf, binary.BigEndian, p); err != nil {
		return false
	}

	r.pmu.Lock()
	r.pending[p.Sequence] = time.Now()
	r.pmu.Unlock()

	_, err := r.conn.WriteToUDP(buf.Bytes(), target)
	if err != nil {
		if st != nil {
			atomic.AddUint64(&st.Dropped, 1)
		}
		return false
	}
	if st != nil {
		atomic.AddUint64(&st.Dispatched, 1)
	}
	r.totalSent.Add(1)
	return true
}

// AckReceived processes an ACK for the given sequence number.
func (r *Router) AckReceived(seq uint64) {
	r.pmu.Lock()
	delete(r.pending, seq)
	r.pmu.Unlock()
	r.totalAck.Add(1)
}

// PendingCount returns the number of unacknowledged packets.
func (r *Router) PendingCount() int {
	r.pmu.Lock()
	defer r.pmu.Unlock()
	return len(r.pending)
}

// TotalSent returns the total number of dispatched packets.
func (r *Router) TotalSent() uint64 {
	return r.totalSent.Load()
}

// TotalAcked returns the total number of acknowledged packets.
func (r *Router) TotalAcked() uint64 {
	return r.totalAck.Load()
}

// Listen handles incoming spike packets and ACKs on the UDP socket.
func (r *Router) Listen() {
	buf := make([]byte, 1024)
	for {
		n, remote, err := r.conn.ReadFromUDP(buf)
		if err != nil {
			continue
		}

		// 8-byte frame = ACK (just the sequence number)
		if n == 8 {
			seq := binary.BigEndian.Uint64(buf[:8])
			r.AckReceived(seq)
			continue
		}

		// Full spike packet
		if n >= PacketSize {
			var p SpikePacket
			if err := binary.Read(bytes.NewReader(buf[:n]), binary.BigEndian, &p); err != nil {
				continue
			}

			// Send ACK back to sender
			ackBuf := make([]byte, 8)
			binary.BigEndian.PutUint64(ackBuf, p.Sequence)
			r.conn.WriteToUDP(ackBuf, remote)
		}
	}
}

// Close shuts down the router's UDP connection.
func (r *Router) Close() error {
	return r.conn.Close()
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

// EncodePacket serializes a SpikePacket to bytes.
func EncodePacket(p SpikePacket) ([]byte, error) {
	buf := new(bytes.Buffer)
	if err := binary.Write(buf, binary.BigEndian, p); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// DecodePacket deserializes a SpikePacket from bytes.
func DecodePacket(data []byte) (SpikePacket, error) {
	var p SpikePacket
	err := binary.Read(bytes.NewReader(data), binary.BigEndian, &p)
	return p, err
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

func main() {
	fmt.Println("SC-NeuroCore AER Interconnect Router active on :9000")
	router := NewRouter("9000")
	go router.Listen()

	select {} // block forever
}
