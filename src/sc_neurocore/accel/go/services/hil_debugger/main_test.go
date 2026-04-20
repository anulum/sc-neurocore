// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — HIL Debugger Tests

package main

import (
	"encoding/json"
	"sync"
	"testing"
)

// ---------------------------------------------------------------------------
// RingBuffer tests
// ---------------------------------------------------------------------------

func TestRingBufferPushAndSnapshot(t *testing.T) {
	rb := NewRingBuffer(4)
	for i := 0; i < 4; i++ {
		rb.Push(SpikeEvent{Sequence: uint64(i), LayerID: "L0"})
	}

	snap := rb.Snapshot(0)
	if len(snap) != 4 {
		t.Fatalf("expected 4 events, got %d", len(snap))
	}
	for i, evt := range snap {
		if evt.Sequence != uint64(i) {
			t.Errorf("event %d: expected seq=%d, got seq=%d", i, i, evt.Sequence)
		}
	}
}

func TestRingBufferOverwrite(t *testing.T) {
	rb := NewRingBuffer(4)
	for i := 0; i < 7; i++ {
		rb.Push(SpikeEvent{Sequence: uint64(i)})
	}
	if rb.Head() != 7 {
		t.Fatalf("head should be 7, got %d", rb.Head())
	}

	snap := rb.Snapshot(0)
	if len(snap) != 4 {
		t.Fatalf("expected 4 events (capacity), got %d", len(snap))
	}
	// Oldest surviving event should be seq=3
	if snap[0].Sequence != 3 {
		t.Errorf("oldest surviving should be seq=3, got seq=%d", snap[0].Sequence)
	}
	if snap[3].Sequence != 6 {
		t.Errorf("newest should be seq=6, got seq=%d", snap[3].Sequence)
	}
}

func TestRingBufferSnapshotN(t *testing.T) {
	rb := NewRingBuffer(16)
	for i := 0; i < 10; i++ {
		rb.Push(SpikeEvent{Sequence: uint64(i)})
	}

	snap := rb.Snapshot(3)
	if len(snap) != 3 {
		t.Fatalf("expected 3 events, got %d", len(snap))
	}
	// Should be the 3 most recent: 7, 8, 9
	if snap[0].Sequence != 7 {
		t.Errorf("expected seq=7, got seq=%d", snap[0].Sequence)
	}
}

func TestRingBufferEmpty(t *testing.T) {
	rb := NewRingBuffer(8)
	snap := rb.Snapshot(0)
	if snap != nil {
		t.Error("empty buffer should return nil")
	}
}

func TestRingBufferConcurrent(t *testing.T) {
	rb := NewRingBuffer(1024)
	var wg sync.WaitGroup

	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(base int) {
			defer wg.Done()
			for i := 0; i < 1000; i++ {
				rb.Push(SpikeEvent{Sequence: uint64(base*1000 + i)})
			}
		}(g)
	}
	wg.Wait()

	if rb.Head() != 8000 {
		t.Errorf("expected 8000 total pushes, got %d", rb.Head())
	}

	snap := rb.Snapshot(0)
	if len(snap) != 1024 {
		t.Errorf("expected 1024 events (capacity), got %d", len(snap))
	}
}

// ---------------------------------------------------------------------------
// SpikeEvent serialization
// ---------------------------------------------------------------------------

func TestSpikeEventJSON(t *testing.T) {
	evt := SpikeEvent{
		Timestamp:   1234567890,
		LayerID:     "L1_V1",
		NeuronID:    42,
		Correlation: 0.05,
		Popcount:    128,
		Precision:   0.97,
		Sequence:    99,
	}

	data, err := json.Marshal(evt)
	if err != nil {
		t.Fatal(err)
	}

	var decoded SpikeEvent
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}

	if decoded.LayerID != "L1_V1" {
		t.Errorf("expected L1_V1, got %s", decoded.LayerID)
	}
	if decoded.NeuronID != 42 {
		t.Errorf("expected 42, got %d", decoded.NeuronID)
	}
	if decoded.Sequence != 99 {
		t.Errorf("expected 99, got %d", decoded.Sequence)
	}
}

// ---------------------------------------------------------------------------
// Hub tests (unit, no websocket)
// ---------------------------------------------------------------------------

func TestHubIngest(t *testing.T) {
	ring := NewRingBuffer(64)
	hub := NewHub(ring)
	go hub.Run()

	for i := 0; i < 10; i++ {
		hub.Ingest(SpikeEvent{Sequence: uint64(i), LayerID: "test"})
	}

	m := hub.GetMetrics()
	if m.EventsReceived != 10 {
		t.Errorf("expected 10 events received, got %d", m.EventsReceived)
	}
	if m.BufferCapacity != 64 {
		t.Errorf("expected capacity=64, got %d", m.BufferCapacity)
	}
}

func TestHubMetricsUptime(t *testing.T) {
	ring := NewRingBuffer(8)
	hub := NewHub(ring)
	go hub.Run()

	m := hub.GetMetrics()
	if m.UptimeSeconds < 0 {
		t.Error("uptime should be non-negative")
	}
	if m.ClientsActive != 0 {
		t.Errorf("expected 0 clients, got %d", m.ClientsActive)
	}
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

func BenchmarkRingBufferPush(b *testing.B) {
	rb := NewRingBuffer(8192)
	evt := SpikeEvent{LayerID: "bench", Popcount: 100}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rb.Push(evt)
	}
}

func BenchmarkRingBufferSnapshot(b *testing.B) {
	rb := NewRingBuffer(8192)
	for i := 0; i < 8192; i++ {
		rb.Push(SpikeEvent{Sequence: uint64(i)})
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = rb.Snapshot(100)
	}
}

func BenchmarkHubIngest(b *testing.B) {
	ring := NewRingBuffer(8192)
	hub := NewHub(ring)
	go hub.Run()
	evt := SpikeEvent{LayerID: "bench", Popcount: 100}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hub.Ingest(evt)
	}
}

// ---------------------------------------------------------------------------
// LayerAggregator tests (Gap 1)
// ---------------------------------------------------------------------------

func TestLayerAggregatorRecord(t *testing.T) {
	la := NewLayerAggregator()
	la.Record(SpikeEvent{LayerID: "L0", Correlation: 0.1, Precision: 0.95})
	la.Record(SpikeEvent{LayerID: "L0", Correlation: 0.3, Precision: 0.85})
	la.Record(SpikeEvent{LayerID: "L1", Correlation: 0.2, Precision: 0.90})

	all := la.All()
	if len(all) != 2 {
		t.Fatalf("expected 2 layers, got %d", len(all))
	}
	l0 := all["L0"]
	if l0.EventCount != 2 {
		t.Errorf("L0 event count: expected 2, got %d", l0.EventCount)
	}
}

func TestLayerStatsMeanCorrelation(t *testing.T) {
	ls := LayerStats{SumCorrelation: 0.4, EventCount: 2}
	if m := ls.MeanCorrelation(); m != 0.2 {
		t.Errorf("expected 0.2, got %f", m)
	}
}

func TestLayerStatsMeanPrecision(t *testing.T) {
	ls := LayerStats{SumPrecision: 1.8, EventCount: 2}
	if m := ls.MeanPrecision(); m != 0.9 {
		t.Errorf("expected 0.9, got %f", m)
	}
}

func TestLayerStatsEmpty(t *testing.T) {
	ls := LayerStats{}
	if ls.MeanCorrelation() != 0 {
		t.Error("empty layer should return 0 mean correlation")
	}
	if ls.MeanPrecision() != 0 {
		t.Error("empty layer should return 0 mean precision")
	}
}

func TestLayerAggregatorGet(t *testing.T) {
	la := NewLayerAggregator()
	la.Record(SpikeEvent{LayerID: "L0", Precision: 0.5})
	ls, ok := la.Get("L0")
	if !ok {
		t.Fatal("expected L0 to be found")
	}
	if ls.EventCount != 1 {
		t.Errorf("expected 1 event, got %d", ls.EventCount)
	}
	_, ok = la.Get("missing")
	if ok {
		t.Error("expected missing layer to not be found")
	}
}

func TestLayerStatsSpikeRate(t *testing.T) {
	ls := LayerStats{EventCount: 100}
	rate := ls.SpikeRate(10.0)
	if rate != 10.0 {
		t.Errorf("expected 10.0, got %f", rate)
	}
	if ls.SpikeRate(0) != 0 {
		t.Error("spike rate with elapsed 0 should be 0")
	}
}

// ---------------------------------------------------------------------------
// ErrorBudget tests (Gap 2)
// ---------------------------------------------------------------------------

func TestErrorBudgetNoViolation(t *testing.T) {
	eb := ErrorBudget{MinPrecision: 0.9, MaxCorrelation: 0.5}
	violated := eb.Check(SpikeEvent{Precision: 0.95, Correlation: 0.1})
	if violated {
		t.Error("should not violate budget")
	}
	if eb.Violations != 0 {
		t.Errorf("expected 0 violations, got %d", eb.Violations)
	}
}

func TestErrorBudgetPrecisionViolation(t *testing.T) {
	eb := ErrorBudget{MinPrecision: 0.9, MaxCorrelation: 0.5}
	violated := eb.Check(SpikeEvent{Precision: 0.8, Correlation: 0.3})
	if !violated {
		t.Error("should violate precision budget")
	}
}

func TestErrorBudgetCorrelationViolation(t *testing.T) {
	eb := ErrorBudget{MinPrecision: 0.9, MaxCorrelation: 0.5}
	violated := eb.Check(SpikeEvent{Precision: 0.95, Correlation: 0.6})
	if !violated {
		t.Error("should violate correlation budget")
	}
}

// ---------------------------------------------------------------------------
// CorrelationWindow tests (Gap 3)
// ---------------------------------------------------------------------------

func TestCorrelationWindowMean(t *testing.T) {
	cw := NewCorrelationWindow(4)
	cw.Add(0.1)
	cw.Add(0.2)
	cw.Add(0.3)
	cw.Add(0.4)
	mean := cw.Mean()
	if mean < 0.24 || mean > 0.26 {
		t.Errorf("expected ~0.25, got %f", mean)
	}
}

func TestCorrelationWindowMax(t *testing.T) {
	cw := NewCorrelationWindow(8)
	cw.Add(0.1)
	cw.Add(0.5)
	cw.Add(0.3)
	if cw.Max() != 0.5 {
		t.Errorf("expected 0.5, got %f", cw.Max())
	}
}

func TestCorrelationWindowEmpty(t *testing.T) {
	cw := NewCorrelationWindow(4)
	if cw.Mean() != 0 {
		t.Error("empty window mean should be 0")
	}
	if cw.Count() != 0 {
		t.Error("empty window count should be 0")
	}
}

func TestCorrelationWindowOverwrite(t *testing.T) {
	cw := NewCorrelationWindow(2)
	cw.Add(0.1)
	cw.Add(0.2)
	cw.Add(0.9) // overwrites 0.1
	if cw.Count() != 2 {
		t.Errorf("expected 2, got %d", cw.Count())
	}
	if cw.Max() != 0.9 {
		t.Errorf("expected max 0.9, got %f", cw.Max())
	}
}

// ---------------------------------------------------------------------------
// PrecisionTracker tests (Gap 4)
// ---------------------------------------------------------------------------

func TestPrecisionTrackerEMA(t *testing.T) {
	pt := NewPrecisionTracker(0.5)
	pt.Update(0.9)
	if pt.EMA != 0.9 {
		t.Errorf("first update should set EMA to value, got %f", pt.EMA)
	}
	pt.Update(0.8)
	expected := 0.5*0.8 + 0.5*0.9 // 0.85
	diff := pt.EMA - expected
	if diff < 0 {
		diff = -diff
	}
	if diff > 1e-10 {
		t.Errorf("expected %f, got %f", expected, pt.EMA)
	}
}

func TestPrecisionTrackerCount(t *testing.T) {
	pt := NewPrecisionTracker(0.1)
	for i := 0; i < 10; i++ {
		pt.Update(0.95)
	}
	if pt.Count != 10 {
		t.Errorf("expected 10, got %d", pt.Count)
	}
}

// ---------------------------------------------------------------------------
// EventFilter tests (Gap 5)
// ---------------------------------------------------------------------------

func TestEventFilterByLayer(t *testing.T) {
	events := []SpikeEvent{
		{LayerID: "L0", NeuronID: 1},
		{LayerID: "L1", NeuronID: 2},
		{LayerID: "L0", NeuronID: 3},
	}
	result := FilterEvents(events, EventFilter{LayerID: "L0"})
	if len(result) != 2 {
		t.Errorf("expected 2, got %d", len(result))
	}
}

func TestEventFilterByNeuron(t *testing.T) {
	events := []SpikeEvent{
		{LayerID: "L0", NeuronID: 5},
		{LayerID: "L0", NeuronID: 15},
		{LayerID: "L0", NeuronID: 25},
	}
	result := FilterEvents(events, EventFilter{MinNeuron: 10, MaxNeuron: 20, HasNeuron: true})
	if len(result) != 1 {
		t.Errorf("expected 1, got %d", len(result))
	}
}

func TestEventFilterMatchAll(t *testing.T) {
	f := EventFilter{}
	if !f.Match(SpikeEvent{LayerID: "any"}) {
		t.Error("empty filter should match everything")
	}
}

// ---------------------------------------------------------------------------
// Trigger tests (Gap 6)
// ---------------------------------------------------------------------------

func TestTriggerNotArmed(t *testing.T) {
	tc := TriggerCondition{MinCorrelation: 0.5, Armed: false}
	if tc.Evaluate(SpikeEvent{Correlation: 1.0}) {
		t.Error("disarmed trigger should not fire")
	}
}

func TestTriggerCorrelation(t *testing.T) {
	tc := TriggerCondition{MinCorrelation: 0.5, Armed: true}
	if !tc.Evaluate(SpikeEvent{Correlation: 0.6}) {
		t.Error("should fire on high correlation")
	}
}

func TestTriggerPrecision(t *testing.T) {
	tc := TriggerCondition{MaxPrecision: 0.8, Armed: true}
	if !tc.Evaluate(SpikeEvent{Precision: 0.7}) {
		t.Error("should fire on low precision")
	}
}

func TestTriggerLog(t *testing.T) {
	tl := TriggerLog{}
	tl.Fire(SpikeEvent{Sequence: 1})
	tl.Fire(SpikeEvent{Sequence: 2})
	if tl.Count() != 2 {
		t.Errorf("expected 2 entries, got %d", tl.Count())
	}
}

// ---------------------------------------------------------------------------
// RateLimiter tests (Gap 7)
// ---------------------------------------------------------------------------

func TestRateLimiterAllow(t *testing.T) {
	rl := NewRateLimiter(3)
	for i := 0; i < 3; i++ {
		if !rl.Allow() {
			t.Errorf("should allow event %d", i)
		}
	}
	if rl.Allow() {
		t.Error("should reject after capacity exhausted")
	}
}

func TestRateLimiterRefill(t *testing.T) {
	rl := NewRateLimiter(2)
	rl.Allow()
	rl.Allow()
	rl.Refill(5)
	if rl.Available() != 2 {
		t.Errorf("expected 2 (capped), got %d", rl.Available())
	}
}

// ---------------------------------------------------------------------------
// HealthCheck tests (Gap 8)
// ---------------------------------------------------------------------------

func TestHealthCheckHealthy(t *testing.T) {
	m := HubMetrics{
		EventsReceived: 100,
		BufferCapacity:  1024,
		BufferHead:      50,
		UptimeSeconds:   10,
	}
	h := CheckHealth(m)
	if h.Status != "healthy" {
		t.Errorf("expected healthy, got %s", h.Status)
	}
	if h.EventsPerSec != 10.0 {
		t.Errorf("expected 10 events/sec, got %f", h.EventsPerSec)
	}
}

func TestHealthCheckBufferPressure(t *testing.T) {
	m := HubMetrics{
		BufferCapacity: 100,
		BufferHead:     100,
		UptimeSeconds:  1,
	}
	h := CheckHealth(m)
	if h.Status != "buffer_pressure" {
		t.Errorf("expected buffer_pressure, got %s", h.Status)
	}
}

// ---------------------------------------------------------------------------
// SnapshotExport tests (Gap 10)
// ---------------------------------------------------------------------------

func TestExportCSV(t *testing.T) {
	events := []SpikeEvent{
		{Timestamp: 1000, LayerID: "L0", NeuronID: 5, Correlation: 0.1, Popcount: 100, Precision: 0.95, Sequence: 0},
	}
	csv := ExportCSV(events)
	if len(csv) == 0 {
		t.Fatal("CSV should not be empty")
	}
	// Check header
	if csv[:9] != "timestamp" {
		t.Error("CSV should start with header")
	}
}

func TestExportJSON(t *testing.T) {
	events := []SpikeEvent{
		{Timestamp: 1000, LayerID: "L0"},
	}
	j, err := ExportJSON(events)
	if err != nil {
		t.Fatal(err)
	}
	if len(j) == 0 {
		t.Fatal("JSON should not be empty")
	}
	var decoded []SpikeEvent
	if err := json.Unmarshal([]byte(j), &decoded); err != nil {
		t.Fatal(err)
	}
	if len(decoded) != 1 {
		t.Errorf("expected 1 event, got %d", len(decoded))
	}
}
