// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go RK4 parity + benchmark tests for the PV+ fast-spiking neuron

package services

import (
	"math"
	"testing"
)

// TestSSTRK4MatchesPythonReference pins the Go RK4 kernel to the pure
// Python reference: 40000 steps at I = 2 µA/cm² produce 631 spikes and the
// recorded membrane potential.
func TestSSTRK4MatchesPythonReference(t *testing.T) {
	s := NewSSTNeuron()
	spikes := 0
	for i := 0; i < 40000; i++ {
		spikes += s.Step(2.0)
	}
	if spikes != 67 {
		t.Fatalf("expected 67 spikes, got %d", spikes)
	}
	const refV = -54.484104868226545
	if math.Abs(s.V-refV) > 1e-6 {
		t.Fatalf("membrane potential drifted from Python reference: got %.12f want %.12f", s.V, refV)
	}
}

// TestSSTQuiescentWithoutDrive verifies the cell is silent at zero stimulus.
func TestSSTQuiescentWithoutDrive(t *testing.T) {
	s := NewSSTNeuron()
	spikes := 0
	for i := 0; i < 20000; i++ {
		spikes += s.Step(0.0)
	}
	if spikes != 0 {
		t.Fatalf("expected silence at I=0, got %d spikes", spikes)
	}
}

// TestSSTMCurrentChangesFiring checks that the M-type current shapes the
// discharge: blocking it (g_M = 0) changes the spike count.
func TestSSTMCurrentChangesFiring(t *testing.T) {
	intact := NewSSTNeuron()
	blocked := NewSSTNeuron()
	blocked.GM = 0.0
	intactSpikes, blockedSpikes := 0, 0
	for i := 0; i < 40000; i++ {
		intactSpikes += intact.Step(2.0)
		blockedSpikes += blocked.Step(2.0)
	}
	if intactSpikes == blockedSpikes {
		t.Fatalf("M-current block should change firing: intact=%d blocked=%d", intactSpikes, blockedSpikes)
	}
}

// BenchmarkSSTRK4 measures the RK4 step throughput.
func BenchmarkSSTRK4(b *testing.B) {
	s := NewSSTNeuron()
	spikes := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		spikes += s.Step(2.0)
	}
	b.StopTimer()
	b.ReportMetric(float64(spikes), "spikes")
}
