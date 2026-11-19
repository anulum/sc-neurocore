// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — AER Interconnect Router

// Package main implements an AER-over-UDP multi-FPGA spike router
// with dynamic routing, ACK-based reliability, sequence tracking,
// and per-route delivery statistics.
//
// Protocol: Each SpikePacket is serialized as a fixed-size big-endian
// binary frame (28 bytes). ACKs are 8-byte sequence echoes.
package main

import (
	"bytes"
	"container/list"
	"context"
	"crypto/rand"
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

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
const pendingLimit = 4096
const ackTimeout = 2 * time.Second
const maxDeliverySequence = 1<<63 - 1

// RouteStats tracks per-route delivery statistics.
type RouteStats struct {
	Dispatched uint64
	Acked      uint64
	Dropped    uint64
}

// pendingAck records the destination actually used for a successful send.
type pendingAck struct {
	upstream uint64
	delivery uint64
	endpoint *net.UDPAddr
	stats    *RouteStats
	deadline time.Time
}

// Router manages route registration, spike dispatch, and ACK tracking.
type Router struct {
	mu           sync.RWMutex
	routes       map[uint32]*net.UDPAddr
	conn         *net.UDPConn
	pending      map[uint64]*list.Element
	wire         map[uint64]*list.Element
	expiry       list.List
	pmu          sync.Mutex
	deliveryBase uint64
	nextDelivery uint64

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
	var generation [8]byte
	_, _ = rand.Read(generation[:])
	return &Router{
		routes:       make(map[uint32]*net.UDPAddr),
		conn:         conn,
		pending:      make(map[uint64]*list.Element),
		wire:         make(map[uint64]*list.Element),
		stats:        make(map[uint32]*RouteStats),
		deliveryBase: binary.BigEndian.Uint64(generation[:]) % maxDeliverySequence,
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

	// Serialise the send and pending insert with ACK processing.
	r.pmu.Lock()
	defer r.pmu.Unlock()
	now := time.Now()
	r.expirePending(now)
	if r.pending[p.Sequence] != nil || len(r.pending) >= pendingLimit || r.nextDelivery == maxDeliverySequence {
		atomic.AddUint64(&st.Dropped, 1)
		return false
	}
	upstream := p.Sequence
	// Rotate the positive sequence space by the process's random starting offset.
	p.Sequence = (r.deliveryBase+r.nextDelivery)%maxDeliverySequence + 1
	r.nextDelivery++
	data, err := EncodePacket(p)
	if err != nil {
		return false
	}
	_, err = r.conn.WriteToUDP(data, target)
	if err != nil {
		atomic.AddUint64(&st.Dropped, 1)
		return false
	}
	entry := r.expiry.PushBack(pendingAck{upstream, p.Sequence, target, st, now.Add(ackTimeout)})
	r.pending[upstream] = entry
	r.wire[p.Sequence] = entry
	atomic.AddUint64(&st.Dispatched, 1)
	r.totalSent.Add(1)
	return true
}

// expirePending retires timed-out sends in deadline order with pmu held.
func (r *Router) expirePending(now time.Time) {
	for entry := r.expiry.Front(); entry != nil; entry = r.expiry.Front() {
		pending := entry.Value.(pendingAck)
		if now.Before(pending.deadline) {
			break
		}
		r.removePending(entry, pending)
		atomic.AddUint64(&pending.stats.Dropped, 1)
	}
}

// removePending deletes both indices for a retired send with pmu held.
func (r *Router) removePending(entry *list.Element, pending pendingAck) {
	delete(r.pending, pending.upstream)
	delete(r.wire, pending.delivery)
	r.expiry.Remove(entry)
}

// AckReceived records a trusted in-process acknowledgement for a live sequence.
func (r *Router) AckReceived(seq uint64) {
	r.ackReceived(seq, nil)
}

// ackReceived matches live UDP ACKs to the endpoint captured at dispatch.
func (r *Router) ackReceived(seq uint64, remote *net.UDPAddr) {
	r.pmu.Lock()
	defer r.pmu.Unlock()
	r.expirePending(time.Now())
	entry := r.pending[seq]
	if remote != nil {
		entry = r.wire[seq]
	}
	if entry == nil {
		return
	}
	pending := entry.Value.(pendingAck)
	peer := pending.endpoint
	if remote != nil && (remote.Port != peer.Port || remote.Zone != peer.Zone || !remote.IP.Equal(peer.IP)) {
		return
	}
	r.removePending(entry, pending)
	atomic.AddUint64(&pending.stats.Acked, 1)
	r.totalAck.Add(1)
}

// PendingCount returns the number of unacknowledged packets.
func (r *Router) PendingCount() int {
	r.pmu.Lock()
	defer r.pmu.Unlock()
	r.expirePending(time.Now())
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
			return
		}

		// 8-byte frame = ACK (just the sequence number)
		if n == 8 {
			seq := binary.BigEndian.Uint64(buf[:8])
			r.ackReceived(seq, remote)
			continue
		}

		// Full spike packet
		if n == PacketSize {
			p, err := DecodePacket(buf[:n])
			if err != nil || !r.DispatchSpike(p) {
				continue
			}

			var ack [8]byte
			binary.BigEndian.PutUint64(ack[:], p.Sequence)
			if _, err := r.conn.WriteToUDP(ack[:], remote); err != nil {
				log.Printf("[router] send ACK: %v", err)
			}
		}
	}
}

// Close shuts down the router's UDP connection.
func (r *Router) Close() error {
	return r.conn.Close()
}

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

func main() {
	port := flag.String("port", "9000", "UDP listen port")
	var routes []string
	flag.Func("route", "repeatable destination ID=HOST:PORT", func(route string) error {
		routes = append(routes, route)
		return nil
	})
	flag.Parse()
	router := NewRouter(*port)
	defer router.Close()
	for _, route := range routes {
		id, address, ok := strings.Cut(route, "=")
		destination, err := strconv.ParseUint(id, 10, 32)
		if !ok || err != nil || address == "" {
			log.Fatalf("[router] invalid route %q, expected ID=HOST:PORT", route)
		}
		if err := router.RegisterRoute(uint32(destination), address); err != nil {
			log.Fatal(err)
		}
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go func() {
		<-ctx.Done()
		_ = router.Close()
	}()
	fmt.Printf("SC-NeuroCore AER Interconnect Router active on %s\n", router.conn.LocalAddr())
	router.Listen()
}
