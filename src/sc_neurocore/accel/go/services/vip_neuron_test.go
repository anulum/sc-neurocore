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

// TestVIPRK4MatchesPythonReference pins the Go RK4 kernel to the pure
// Python reference: 40000 steps at I = 2 µA/cm² produce 631 spikes and the
// recorded membrane potential.
func TestVIPRK4MatchesPythonReference(t *testing.T) {
	s := NewVIPNeuron()
	spikes := 0
	for i := 0; i < 40000; i++ {
		spikes += s.Step(1.0)
	}
	if spikes != 36 {
		t.Fatalf("expected 36 spikes, got %d", spikes)
	}
	const refV = -57.91028344798147
	if math.Abs(s.V-refV) > 1e-6 {
		t.Fatalf("membrane potential drifted from Python reference: got %.12f want %.12f", s.V, refV)
	}
}

// TestVIPQuiescentWithoutDrive verifies the cell is silent at zero stimulus.
func TestVIPQuiescentWithoutDrive(t *testing.T) {
	s := NewVIPNeuron()
	spikes := 0
	for i := 0; i < 20000; i++ {
		spikes += s.Step(0.0)
	}
	if spikes != 0 {
		t.Fatalf("expected silence at I=0, got %d spikes", spikes)
	}
}

// TestVIPACurrentChangesFiring checks that the A-type current shapes the
// discharge: blocking it (g_A = 0) changes the spike count.
func TestVIPACurrentChangesFiring(t *testing.T) {
	intact := NewVIPNeuron()
	blocked := NewVIPNeuron()
	blocked.GA = 0.0
	intactSpikes, blockedSpikes := 0, 0
	for i := 0; i < 40000; i++ {
		intactSpikes += intact.Step(1.0)
		blockedSpikes += blocked.Step(1.0)
	}
	if intactSpikes == blockedSpikes {
		t.Fatalf("A-current block should change firing: intact=%d blocked=%d", intactSpikes, blockedSpikes)
	}
}

// BenchmarkVIPRK4 measures the RK4 step throughput.
func BenchmarkVIPRK4(b *testing.B) {
	s := NewVIPNeuron()
	spikes := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		spikes += s.Step(1.0)
	}
	b.StopTimer()
	b.ReportMetric(float64(spikes), "spikes")
}
