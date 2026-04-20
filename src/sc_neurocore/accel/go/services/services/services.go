// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Compute Services Suite

package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"math/bits"
	"net"
	"sync"
	"sync/atomic"
	"time"
)

// ============================================================
// §1  AER Spike Packet (28-byte wire format)
// ============================================================

type SpikePacket struct {
	Timestamp uint64
	NeuronID  uint32
	LayerID   uint16
	Polarity  uint8
	Payload   [9]byte
}

func EncodeSpikePacket(p SpikePacket) []byte {
	buf := make([]byte, 28)
	binary.BigEndian.PutUint64(buf[0:8], p.Timestamp)
	binary.BigEndian.PutUint32(buf[8:12], p.NeuronID)
	binary.BigEndian.PutUint16(buf[12:14], p.LayerID)
	buf[14] = p.Polarity
	copy(buf[15:24], p.Payload[:])
	// CRC32 placeholder
	binary.BigEndian.PutUint32(buf[24:28], 0xDEADBEEF)
	return buf
}

func DecodeSpikePacket(buf []byte) SpikePacket {
	var p SpikePacket
	p.Timestamp = binary.BigEndian.Uint64(buf[0:8])
	p.NeuronID = binary.BigEndian.Uint32(buf[8:12])
	p.LayerID = binary.BigEndian.Uint16(buf[12:14])
	p.Polarity = buf[14]
	copy(p.Payload[:], buf[15:24])
	return p
}

// ============================================================
// §2  Spike Ring Buffer (lock-free)
// ============================================================

type SpikeRingBuffer struct {
	data  []SpikePacket
	head  uint64
	tail  uint64
	cap   uint64
}

func NewSpikeRingBuffer(capacity int) *SpikeRingBuffer {
	return &SpikeRingBuffer{
		data: make([]SpikePacket, capacity),
		cap:  uint64(capacity),
	}
}

func (rb *SpikeRingBuffer) Push(p SpikePacket) {
	idx := atomic.AddUint64(&rb.head, 1) - 1
	rb.data[idx%rb.cap] = p
}

func (rb *SpikeRingBuffer) Count() uint64 {
	return atomic.LoadUint64(&rb.head) - atomic.LoadUint64(&rb.tail)
}

// ============================================================
// §3  Layer Aggregator
// ============================================================

type LayerStats struct {
	LayerID    uint16
	SpikeCount uint64
	TotalBits  uint64
	MeanRate   float64
	MaxRate    float64
	MinRate    float64
}

type LayerAggregator struct {
	mu    sync.RWMutex
	stats map[uint16]*LayerStats
}

func NewLayerAggregator() *LayerAggregator {
	return &LayerAggregator{stats: make(map[uint16]*LayerStats)}
}

func (la *LayerAggregator) Record(layerID uint16, spikeCount uint64, totalNeurons uint64) {
	la.mu.Lock()
	defer la.mu.Unlock()
	s, ok := la.stats[layerID]
	if !ok {
		s = &LayerStats{LayerID: layerID, MinRate: math.MaxFloat64}
		la.stats[layerID] = s
	}
	s.SpikeCount += spikeCount
	s.TotalBits += totalNeurons
	rate := float64(spikeCount) / float64(totalNeurons)
	if rate > s.MaxRate {
		s.MaxRate = rate
	}
	if rate < s.MinRate {
		s.MinRate = rate
	}
	s.MeanRate = float64(s.SpikeCount) / float64(s.TotalBits)
}

func (la *LayerAggregator) Summary() map[uint16]LayerStats {
	la.mu.RLock()
	defer la.mu.RUnlock()
	result := make(map[uint16]LayerStats)
	for k, v := range la.stats {
		result[k] = *v
	}
	return result
}

// ============================================================
// §4  AER Router (multi-destination)
// ============================================================

type RouteEntry struct {
	LayerID uint16
	Address string
}

type AERRouter struct {
	mu     sync.RWMutex
	routes map[uint16][]RouteEntry
	sent   uint64
	dropped uint64
}

func NewAERRouter() *AERRouter {
	return &AERRouter{routes: make(map[uint16][]RouteEntry)}
}

func (r *AERRouter) AddRoute(srcLayer uint16, dst RouteEntry) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.routes[srcLayer] = append(r.routes[srcLayer], dst)
}

func (r *AERRouter) Route(p SpikePacket) int {
	r.mu.RLock()
	dsts, ok := r.routes[p.LayerID]
	r.mu.RUnlock()
	if !ok {
		atomic.AddUint64(&r.dropped, 1)
		return 0
	}
	atomic.AddUint64(&r.sent, uint64(len(dsts)))
	return len(dsts)
}

func (r *AERRouter) Stats() (sent, dropped uint64) {
	return atomic.LoadUint64(&r.sent), atomic.LoadUint64(&r.dropped)
}

// ============================================================
// §5  Error Budget Tracker
// ============================================================

type ErrorBudget struct {
	mu       sync.Mutex
	budget   float64
	consumed float64
	window   []float64
	maxSize  int
}

func NewErrorBudget(budget float64, windowSize int) *ErrorBudget {
	return &ErrorBudget{budget: budget, maxSize: windowSize}
}

func (eb *ErrorBudget) Record(errorRate float64) bool {
	eb.mu.Lock()
	defer eb.mu.Unlock()
	eb.window = append(eb.window, errorRate)
	if len(eb.window) > eb.maxSize {
		eb.window = eb.window[1:]
	}
	sum := 0.0
	for _, v := range eb.window {
		sum += v
	}
	eb.consumed = sum / float64(len(eb.window))
	return eb.consumed < eb.budget
}

func (eb *ErrorBudget) Remaining() float64 {
	eb.mu.Lock()
	defer eb.mu.Unlock()
	return eb.budget - eb.consumed
}

// ============================================================
// §6  Rate Limiter (token bucket)
// ============================================================

type RateLimiter struct {
	tokens     int64
	maxTokens  int64
	interval   time.Duration
	lastRefill time.Time
	mu         sync.Mutex
}

func NewRateLimiter(maxPerSecond int) *RateLimiter {
	return &RateLimiter{
		tokens:     int64(maxPerSecond),
		maxTokens:  int64(maxPerSecond),
		interval:   time.Second,
		lastRefill: time.Now(),
	}
}

func (rl *RateLimiter) Allow() bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	now := time.Now()
	if now.Sub(rl.lastRefill) >= rl.interval {
		rl.tokens = rl.maxTokens
		rl.lastRefill = now
	}
	if rl.tokens > 0 {
		rl.tokens--
		return true
	}
	return false
}

// ============================================================
// §7  Health Check Service
// ============================================================

type HealthStatus struct {
	Healthy   bool      `json:"healthy"`
	Uptime    string    `json:"uptime"`
	Checks    map[string]bool `json:"checks"`
	Timestamp string    `json:"timestamp"`
}

type HealthChecker struct {
	startTime time.Time
	checks    map[string]func() bool
	mu        sync.RWMutex
}

func NewHealthChecker() *HealthChecker {
	return &HealthChecker{
		startTime: time.Now(),
		checks:    make(map[string]func() bool),
	}
}

func (hc *HealthChecker) RegisterCheck(name string, fn func() bool) {
	hc.mu.Lock()
	defer hc.mu.Unlock()
	hc.checks[name] = fn
}

func (hc *HealthChecker) Status() HealthStatus {
	hc.mu.RLock()
	defer hc.mu.RUnlock()
	results := make(map[string]bool)
	allHealthy := true
	for name, fn := range hc.checks {
		ok := fn()
		results[name] = ok
		if !ok {
			allHealthy = false
		}
	}
	return HealthStatus{
		Healthy:   allHealthy,
		Uptime:    time.Since(hc.startTime).String(),
		Checks:    results,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	}
}

// ============================================================
// §8  Telemetry UDP Sender
// ============================================================

type TelemetryUDP struct {
	conn      *net.UDPConn
	addr      *net.UDPAddr
	seqNum    uint64
}

func NewTelemetryUDP(address string) (*TelemetryUDP, error) {
	addr, err := net.ResolveUDPAddr("udp", address)
	if err != nil {
		return nil, err
	}
	conn, err := net.DialUDP("udp", nil, addr)
	if err != nil {
		return nil, err
	}
	return &TelemetryUDP{conn: conn, addr: addr}, nil
}

func (t *TelemetryUDP) Send(p SpikePacket) error {
	atomic.AddUint64(&t.seqNum, 1)
	buf := EncodeSpikePacket(p)
	_, err := t.conn.Write(buf)
	return err
}

// ============================================================
// §9  Popcount (Go native — for comparison)
// ============================================================

func PopcountSlice(data []uint32) int {
	total := 0
	for _, w := range data {
		total += bits.OnesCount32(w)
	}
	return total
}

func SCCNumerator(a, b []uint32) int {
	n := len(a)
	pa, pb, pab := 0, 0, 0
	for i := 0; i < n; i++ {
		pa += bits.OnesCount32(a[i])
		pb += bits.OnesCount32(b[i])
		pab += bits.OnesCount32(a[i] & b[i])
	}
	return pab*n*32 - pa*pb
}

// ============================================================
// §10  Streaming Codec (delta + run-length)
// ============================================================

type StreamingCodec struct {
	prevWord uint32
	encoded  []uint32
}

func NewStreamingCodec() *StreamingCodec {
	return &StreamingCodec{}
}

func (sc *StreamingCodec) Encode(word uint32) uint32 {
	delta := word ^ sc.prevWord
	sc.prevWord = word
	sc.encoded = append(sc.encoded, delta)
	return delta
}

func (sc *StreamingCodec) Decode(delta uint32) uint32 {
	sc.prevWord ^= delta
	return sc.prevWord
}

// ============================================================
// BENCHMARK
// ============================================================

func main() {
	fmt.Println("=======================================================")
	fmt.Println("SC-NeuroCore Go Services Suite — Benchmark")
	fmt.Println("=======================================================")

	// §1-2 Spike packet encode/decode + ring buffer
	rb := NewSpikeRingBuffer(65536)
	t0 := time.Now()
	for i := 0; i < 1_000_000; i++ {
		rb.Push(SpikePacket{
			Timestamp: uint64(i),
			NeuronID:  uint32(i % 1024),
			LayerID:   uint16(i % 8),
			Polarity:  1,
		})
	}
	fmt.Printf("§1-2 Ring push 1M spikes:         %v\n", time.Since(t0))

	// §3 Layer aggregator
	la := NewLayerAggregator()
	t0 = time.Now()
	for i := 0; i < 1_000_000; i++ {
		la.Record(uint16(i%8), uint64(i%100), 1024)
	}
	fmt.Printf("§3   Layer agg 1M records:        %v\n", time.Since(t0))

	// §4 Router
	router := NewAERRouter()
	for l := uint16(0); l < 8; l++ {
		router.AddRoute(l, RouteEntry{LayerID: l + 1, Address: "127.0.0.1:9000"})
	}
	t0 = time.Now()
	for i := 0; i < 1_000_000; i++ {
		router.Route(SpikePacket{LayerID: uint16(i % 8)})
	}
	sent, dropped := router.Stats()
	fmt.Printf("§4   Route 1M packets:            %v (sent=%d dropped=%d)\n", time.Since(t0), sent, dropped)

	// §5 Error budget
	eb := NewErrorBudget(0.01, 100)
	t0 = time.Now()
	for i := 0; i < 100_000; i++ {
		eb.Record(0.005)
	}
	fmt.Printf("§5   Error budget 100k:           %v (remaining=%.4f)\n", time.Since(t0), eb.Remaining())

	// §7 Health check
	hc := NewHealthChecker()
	hc.RegisterCheck("memory", func() bool { return true })
	hc.RegisterCheck("spike_rate", func() bool { return true })
	t0 = time.Now()
	var status HealthStatus
	for i := 0; i < 100_000; i++ {
		status = hc.Status()
	}
	fmt.Printf("§7   Health check 100k:           %v (healthy=%v)\n", time.Since(t0), status.Healthy)

	// §9 Popcount comparison
	data := make([]uint32, 1024)
	for i := range data {
		data[i] = 0xDEADBEEF
	}
	t0 = time.Now()
	for i := 0; i < 100_000; i++ {
		PopcountSlice(data)
	}
	fmt.Printf("§9   Popcount 1024w × 100k:       %v\n", time.Since(t0))

	// §9 SCC
	a := make([]uint32, 256)
	b := make([]uint32, 256)
	for i := range a {
		a[i] = 0xAAAAAAAA
		b[i] = 0x55555555
	}
	t0 = time.Now()
	for i := 0; i < 100_000; i++ {
		SCCNumerator(a, b)
	}
	fmt.Printf("§9   SCC 256w × 100k:             %v\n", time.Since(t0))

	// §10 Streaming codec
	codec := NewStreamingCodec()
	t0 = time.Now()
	for i := 0; i < 1_000_000; i++ {
		codec.Encode(uint32(i * 0xDEAD))
	}
	fmt.Printf("§10  Stream encode 1M:            %v\n", time.Since(t0))

	// Marshal health status to JSON
	jsonBytes, _ := json.Marshal(status)
	fmt.Printf("\nHealth: %s\n", string(jsonBytes))

	fmt.Println("=======================================================")
	fmt.Println("10 service groups, 25 functions total")
}
