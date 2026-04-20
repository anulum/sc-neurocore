// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — HIL Debugger Telemetry Server

// Package main implements a Hardware-in-the-Loop (HIL) debugger with
// real-time bitstream telemetry, a lock-free ring buffer, WebSocket
// broadcast hub, and an HTTP metrics endpoint.
//
// Architecture:
//
//	FPGA/Sim → [SpikeEvent] → RingBuffer → Hub → WebSocket clients
//	                                         ↓
//	                                     /metrics (JSON)
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
)

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

// SpikeEvent represents a single telemetry sample from an FPGA or simulator.
type SpikeEvent struct {
	Timestamp   int64   `json:"ts"`
	LayerID     string  `json:"layer_id"`
	NeuronID    uint32  `json:"neuron_id"`
	Correlation float64 `json:"correlation"`
	Popcount    uint32  `json:"popcount"`
	Precision   float64 `json:"precision"`
	Sequence    uint64  `json:"seq"`
}

// HubMetrics tracks aggregate statistics exposed via /metrics.
type HubMetrics struct {
	EventsReceived  uint64 `json:"events_received"`
	EventsBroadcast uint64 `json:"events_broadcast"`
	ClientsActive   int    `json:"clients_active"`
	BufferCapacity  int    `json:"buffer_capacity"`
	BufferHead      uint64 `json:"buffer_head"`
	UptimeSeconds   int64  `json:"uptime_seconds"`
}

// ---------------------------------------------------------------------------
// RingBuffer — lock-free overwrite-on-full telemetry ring
// ---------------------------------------------------------------------------

// RingBuffer is a fixed-capacity circular buffer for SpikeEvent telemetry.
// Writers atomically advance the head; readers snapshot the buffer contents.
// On overflow, the oldest entry is silently overwritten (no allocation).
type RingBuffer struct {
	data []SpikeEvent
	cap  int
	head atomic.Uint64
	mu   sync.RWMutex
}

// NewRingBuffer creates a ring buffer with the given capacity.
func NewRingBuffer(capacity int) *RingBuffer {
	if capacity <= 0 {
		capacity = 1024
	}
	return &RingBuffer{
		data: make([]SpikeEvent, capacity),
		cap:  capacity,
	}
}

// Push appends an event, overwriting the oldest if full.
func (rb *RingBuffer) Push(evt SpikeEvent) {
	idx := rb.head.Add(1) - 1
	rb.mu.Lock()
	rb.data[int(idx%uint64(rb.cap))] = evt
	rb.mu.Unlock()
}

// Snapshot returns the most recent `n` events in chronological order.
func (rb *RingBuffer) Snapshot(n int) []SpikeEvent {
	head := rb.head.Load()
	if head == 0 {
		return nil
	}
	count := int(head)
	if count > rb.cap {
		count = rb.cap
	}
	if n > 0 && n < count {
		count = n
	}

	result := make([]SpikeEvent, count)
	rb.mu.RLock()
	for i := 0; i < count; i++ {
		idx := (int(head) - count + i) % rb.cap
		if idx < 0 {
			idx += rb.cap
		}
		result[i] = rb.data[idx]
	}
	rb.mu.RUnlock()
	return result
}

// Head returns the total number of events ever pushed.
func (rb *RingBuffer) Head() uint64 {
	return rb.head.Load()
}

// Cap returns the buffer capacity.
func (rb *RingBuffer) Cap() int {
	return rb.cap
}

// ---------------------------------------------------------------------------
// WebSocket Hub — broadcast with backpressure
// ---------------------------------------------------------------------------

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

// Hub manages WebSocket client connections and broadcasts SpikeEvents.
type Hub struct {
	clients    map[*websocket.Conn]bool
	broadcast  chan []byte
	register   chan *websocket.Conn
	unregister chan *websocket.Conn
	mu         sync.Mutex

	ring    *RingBuffer
	metrics HubMetrics
	started time.Time
}

// NewHub creates a hub backed by the given ring buffer.
func NewHub(ring *RingBuffer) *Hub {
	return &Hub{
		broadcast:  make(chan []byte, 256),
		register:   make(chan *websocket.Conn),
		unregister: make(chan *websocket.Conn),
		clients:    make(map[*websocket.Conn]bool),
		ring:       ring,
		started:    time.Now(),
	}
}

// Run processes hub events in a dedicated goroutine.
func (h *Hub) Run() {
	for {
		select {
		case client := <-h.register:
			h.mu.Lock()
			h.clients[client] = true
			h.mu.Unlock()
			log.Printf("[hub] client connected, total=%d", len(h.clients))
		case client := <-h.unregister:
			h.mu.Lock()
			if _, ok := h.clients[client]; ok {
				delete(h.clients, client)
				client.Close()
			}
			h.mu.Unlock()
			log.Printf("[hub] client disconnected, total=%d", len(h.clients))
		case message := <-h.broadcast:
			h.mu.Lock()
			for client := range h.clients {
				err := client.WriteMessage(websocket.TextMessage, message)
				if err != nil {
					client.Close()
					delete(h.clients, client)
				}
			}
			atomic.AddUint64(&h.metrics.EventsBroadcast, 1)
			h.mu.Unlock()
		}
	}
}

// Ingest records an event into the ring buffer and broadcasts it.
func (h *Hub) Ingest(evt SpikeEvent) {
	h.ring.Push(evt)
	atomic.AddUint64(&h.metrics.EventsReceived, 1)

	msg, err := json.Marshal(evt)
	if err != nil {
		return
	}
	select {
	case h.broadcast <- msg:
	default:
		// Drop if broadcast channel is full (backpressure)
	}
}

// GetMetrics returns a snapshot of current hub metrics.
func (h *Hub) GetMetrics() HubMetrics {
	h.mu.Lock()
	m := HubMetrics{
		EventsReceived:  atomic.LoadUint64(&h.metrics.EventsReceived),
		EventsBroadcast: atomic.LoadUint64(&h.metrics.EventsBroadcast),
		ClientsActive:   len(h.clients),
		BufferCapacity:  h.ring.Cap(),
		BufferHead:      h.ring.Head(),
		UptimeSeconds:   int64(time.Since(h.started).Seconds()),
	}
	h.mu.Unlock()
	return m
}

// ---------------------------------------------------------------------------
// LayerStats — per-layer running aggregation (Gap 1)
// ---------------------------------------------------------------------------

// LayerStats tracks running statistics for a single neural layer.
type LayerStats struct {
	LayerID        string  `json:"layer_id"`
	EventCount     uint64  `json:"event_count"`
	SumCorrelation float64 `json:"sum_correlation"`
	SumPrecision   float64 `json:"sum_precision"`
	SumPopcount    uint64  `json:"sum_popcount"`
	MinPrecision   float64 `json:"min_precision"`
	MaxCorrelation float64 `json:"max_correlation"`
}

// MeanCorrelation returns the average correlation for this layer.
func (ls *LayerStats) MeanCorrelation() float64 {
	if ls.EventCount == 0 {
		return 0
	}
	return ls.SumCorrelation / float64(ls.EventCount)
}

// MeanPrecision returns the average effective precision.
func (ls *LayerStats) MeanPrecision() float64 {
	if ls.EventCount == 0 {
		return 0
	}
	return ls.SumPrecision / float64(ls.EventCount)
}

// SpikeRate returns events per second given elapsed duration.
func (ls *LayerStats) SpikeRate(elapsedSec float64) float64 {
	if elapsedSec <= 0 {
		return 0
	}
	return float64(ls.EventCount) / elapsedSec
}

// LayerAggregator collects per-layer statistics.
type LayerAggregator struct {
	layers map[string]*LayerStats
	mu     sync.RWMutex
}

// NewLayerAggregator creates a new aggregator.
func NewLayerAggregator() *LayerAggregator {
	return &LayerAggregator{
		layers: make(map[string]*LayerStats),
	}
}

// Record adds a spike event to the appropriate layer's stats.
func (la *LayerAggregator) Record(evt SpikeEvent) {
	la.mu.Lock()
	defer la.mu.Unlock()
	ls, ok := la.layers[evt.LayerID]
	if !ok {
		ls = &LayerStats{
			LayerID:      evt.LayerID,
			MinPrecision: evt.Precision,
		}
		la.layers[evt.LayerID] = ls
	}
	ls.EventCount++
	ls.SumCorrelation += evt.Correlation
	ls.SumPrecision += evt.Precision
	ls.SumPopcount += uint64(evt.Popcount)
	if evt.Precision < ls.MinPrecision {
		ls.MinPrecision = evt.Precision
	}
	if evt.Correlation > ls.MaxCorrelation {
		ls.MaxCorrelation = evt.Correlation
	}
}

// All returns a snapshot of all layer stats.
func (la *LayerAggregator) All() map[string]LayerStats {
	la.mu.RLock()
	defer la.mu.RUnlock()
	result := make(map[string]LayerStats, len(la.layers))
	for k, v := range la.layers {
		result[k] = *v
	}
	return result
}

// Get returns stats for a specific layer.
func (la *LayerAggregator) Get(layerID string) (LayerStats, bool) {
	la.mu.RLock()
	defer la.mu.RUnlock()
	ls, ok := la.layers[layerID]
	if !ok {
		return LayerStats{}, false
	}
	return *ls, true
}

// ---------------------------------------------------------------------------
// ErrorBudget — threshold-based alerting (Gap 2)
// ---------------------------------------------------------------------------

// ErrorBudget tracks whether per-layer precision stays within budget.
type ErrorBudget struct {
	MinPrecision   float64 `json:"min_precision"`
	MaxCorrelation float64 `json:"max_correlation"`
	Violations     uint64  `json:"violations"`
}

// Check evaluates an event against the error budget. Returns true if violated.
func (eb *ErrorBudget) Check(evt SpikeEvent) bool {
	violated := false
	if evt.Precision < eb.MinPrecision {
		violated = true
	}
	if evt.Correlation > eb.MaxCorrelation {
		violated = true
	}
	if violated {
		eb.Violations++
	}
	return violated
}

// ---------------------------------------------------------------------------
// CorrelationWindow — sliding window SCC (Gap 3)
// ---------------------------------------------------------------------------

// CorrelationWindow maintains a sliding window of correlation values.
type CorrelationWindow struct {
	values []float64
	cap    int
	pos    int
	full   bool
}

// NewCorrelationWindow creates a window of the given size.
func NewCorrelationWindow(size int) *CorrelationWindow {
	if size <= 0 {
		size = 128
	}
	return &CorrelationWindow{
		values: make([]float64, size),
		cap:    size,
	}
}

// Add appends a correlation sample.
func (cw *CorrelationWindow) Add(v float64) {
	cw.values[cw.pos] = v
	cw.pos = (cw.pos + 1) % cw.cap
	if cw.pos == 0 {
		cw.full = true
	}
}

// Mean returns the mean of the window.
func (cw *CorrelationWindow) Mean() float64 {
	n := cw.count()
	if n == 0 {
		return 0
	}
	sum := 0.0
	for i := 0; i < n; i++ {
		sum += cw.values[i]
	}
	return sum / float64(n)
}

// Max returns the maximum correlation in the window.
func (cw *CorrelationWindow) Max() float64 {
	n := cw.count()
	if n == 0 {
		return 0
	}
	m := cw.values[0]
	for i := 1; i < n; i++ {
		if cw.values[i] > m {
			m = cw.values[i]
		}
	}
	return m
}

func (cw *CorrelationWindow) count() int {
	if cw.full {
		return cw.cap
	}
	return cw.pos
}

// Count returns the number of samples in the window.
func (cw *CorrelationWindow) Count() int {
	return cw.count()
}

// ---------------------------------------------------------------------------
// PrecisionTracker — rolling effective precision (Gap 4)
// ---------------------------------------------------------------------------

// PrecisionTracker maintains an exponential moving average of precision.
type PrecisionTracker struct {
	EMA   float64 `json:"ema"`
	Alpha float64 `json:"alpha"`
	Count uint64  `json:"count"`
}

// NewPrecisionTracker creates a tracker with smoothing factor alpha.
func NewPrecisionTracker(alpha float64) *PrecisionTracker {
	if alpha <= 0 || alpha > 1 {
		alpha = 0.05
	}
	return &PrecisionTracker{Alpha: alpha}
}

// Update adds a precision sample.
func (pt *PrecisionTracker) Update(precision float64) {
	pt.Count++
	if pt.Count == 1 {
		pt.EMA = precision
		return
	}
	pt.EMA = pt.Alpha*precision + (1-pt.Alpha)*pt.EMA
}

// ---------------------------------------------------------------------------
// EventFilter — query by layer/neuron range (Gap 5)
// ---------------------------------------------------------------------------

// EventFilter selects events matching criteria.
type EventFilter struct {
	LayerID    string `json:"layer_id,omitempty"`
	MinNeuron  int    `json:"min_neuron,omitempty"`
	MaxNeuron  int    `json:"max_neuron,omitempty"`
	HasNeuron  bool   `json:"-"`
}

// Match returns true if the event passes the filter.
func (ef *EventFilter) Match(evt SpikeEvent) bool {
	if ef.LayerID != "" && evt.LayerID != ef.LayerID {
		return false
	}
	if ef.HasNeuron {
		nid := int(evt.NeuronID)
		if nid < ef.MinNeuron || nid > ef.MaxNeuron {
			return false
		}
	}
	return true
}

// FilterEvents applies a filter to a slice of events.
func FilterEvents(events []SpikeEvent, f EventFilter) []SpikeEvent {
	var result []SpikeEvent
	for _, evt := range events {
		if f.Match(evt) {
			result = append(result, evt)
		}
	}
	return result
}

// ---------------------------------------------------------------------------
// Trigger — conditional breakpoint (Gap 6)
// ---------------------------------------------------------------------------

// TriggerCondition defines when the debugger should fire.
type TriggerCondition struct {
	MinCorrelation float64 `json:"min_correlation"`
	MaxPrecision   float64 `json:"max_precision"`
	LayerID        string  `json:"layer_id,omitempty"`
	Armed          bool    `json:"armed"`
}

// Evaluate checks if an event trips the trigger.
func (tc *TriggerCondition) Evaluate(evt SpikeEvent) bool {
	if !tc.Armed {
		return false
	}
	if tc.LayerID != "" && evt.LayerID != tc.LayerID {
		return false
	}
	if evt.Correlation >= tc.MinCorrelation && tc.MinCorrelation > 0 {
		return true
	}
	if evt.Precision <= tc.MaxPrecision && tc.MaxPrecision > 0 {
		return true
	}
	return false
}

// TriggerLog records fired triggers for post-mortem.
type TriggerLog struct {
	Entries []SpikeEvent
	mu      sync.Mutex
}

// Fire records a triggered event.
func (tl *TriggerLog) Fire(evt SpikeEvent) {
	tl.mu.Lock()
	tl.Entries = append(tl.Entries, evt)
	tl.mu.Unlock()
}

// Count returns fired trigger count.
func (tl *TriggerLog) Count() int {
	tl.mu.Lock()
	defer tl.mu.Unlock()
	return len(tl.Entries)
}

// ---------------------------------------------------------------------------
// RateLimiter — token bucket for high-speed streams (Gap 7)
// ---------------------------------------------------------------------------

// RateLimiter implements a simple token-bucket rate limiter.
type RateLimiter struct {
	tokens   int64
	capacity int64
	mu       sync.Mutex
}

// NewRateLimiter creates a limiter with the given capacity.
func NewRateLimiter(capacity int64) *RateLimiter {
	return &RateLimiter{tokens: capacity, capacity: capacity}
}

// Allow returns true if an event can be processed, consuming one token.
func (rl *RateLimiter) Allow() bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	if rl.tokens > 0 {
		rl.tokens--
		return true
	}
	return false
}

// Refill adds tokens back up to capacity.
func (rl *RateLimiter) Refill(n int64) {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	rl.tokens += n
	if rl.tokens > rl.capacity {
		rl.tokens = rl.capacity
	}
}

// Available returns current token count.
func (rl *RateLimiter) Available() int64 {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	return rl.tokens
}

// ---------------------------------------------------------------------------
// HealthCheck (Gap 8)
// ---------------------------------------------------------------------------

// HealthStatus represents the debugger health.
type HealthStatus struct {
	Status        string `json:"status"`
	EventsPerSec  float64 `json:"events_per_sec"`
	BufferUsage   float64 `json:"buffer_usage"`
	ClientsActive int     `json:"clients_active"`
}

// CheckHealth computes health status from hub metrics.
func CheckHealth(m HubMetrics) HealthStatus {
	usage := 0.0
	if m.BufferCapacity > 0 {
		used := m.BufferHead
		if used > uint64(m.BufferCapacity) {
			used = uint64(m.BufferCapacity)
		}
		usage = float64(used) / float64(m.BufferCapacity)
	}
	eps := 0.0
	if m.UptimeSeconds > 0 {
		eps = float64(m.EventsReceived) / float64(m.UptimeSeconds)
	}
	status := "healthy"
	if usage > 0.95 {
		status = "buffer_pressure"
	}
	return HealthStatus{
		Status:        status,
		EventsPerSec:  eps,
		BufferUsage:   usage,
		ClientsActive: m.ClientsActive,
	}
}

// ---------------------------------------------------------------------------
// SnapshotExport — CSV/JSON file export (Gap 10)
// ---------------------------------------------------------------------------

// ExportCSV converts events to CSV string.
func ExportCSV(events []SpikeEvent) string {
	header := "timestamp,layer_id,neuron_id,correlation,popcount,precision,sequence\n"
	body := ""
	for _, e := range events {
		body += fmt.Sprintf("%d,%s,%d,%.6f,%d,%.6f,%d\n",
			e.Timestamp, e.LayerID, e.NeuronID,
			e.Correlation, e.Popcount, e.Precision, e.Sequence)
	}
	return header + body
}

// ExportJSON converts events to a JSON array string.
func ExportJSON(events []SpikeEvent) (string, error) {
	data, err := json.MarshalIndent(events, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// ---------------------------------------------------------------------------
// HTTP handlers
// ---------------------------------------------------------------------------

func serveWs(hub *Hub, w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("[ws] upgrade error:", err)
		return
	}
	hub.register <- conn

	go func() {
		defer func() { hub.unregister <- conn }()
		for {
			_, _, err := conn.ReadMessage()
			if err != nil {
				break
			}
		}
	}()
}

func serveMetrics(hub *Hub, w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	m := hub.GetMetrics()
	json.NewEncoder(w).Encode(m)
}

func serveHistory(hub *Hub, w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	events := hub.ring.Snapshot(100)
	json.NewEncoder(w).Encode(events)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

func main() {
	ring := NewRingBuffer(8192)
	hub := NewHub(ring)
	go hub.Run()

	// Simulated FPGA data source (replace with real AER ingest)
	go func() {
		var seq uint64
		layers := []string{"L0_Retina", "L1_V1", "L2_V2", "L3_IT", "L4_Motor"}
		for {
			layer := layers[rand.Intn(len(layers))]
			evt := SpikeEvent{
				Timestamp:   time.Now().UnixMilli(),
				LayerID:     layer,
				NeuronID:    uint32(rand.Intn(1024)),
				Correlation: rand.Float64() * 0.1,
				Popcount:    uint32(rand.Intn(256)),
				Precision:   0.95 + rand.Float64()*0.05,
				Sequence:    seq,
			}
			hub.Ingest(evt)
			seq++
			time.Sleep(10 * time.Millisecond) // 100 Hz
		}
	}()

	http.HandleFunc("/ws", func(w http.ResponseWriter, r *http.Request) {
		serveWs(hub, w, r)
	})
	http.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		serveMetrics(hub, w, r)
	})
	http.HandleFunc("/history", func(w http.ResponseWriter, r *http.Request) {
		serveHistory(hub, w, r)
	})
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		h := CheckHealth(hub.GetMetrics())
		json.NewEncoder(w).Encode(h)
	})

	port := os.Getenv("HIL_PORT")
	if port == "" {
		port = "8081"
	}

	fmt.Printf("SC-NeuroCore HIL Debugger: active on :%s (/ws /metrics /history /health)\n", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}
