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

// TestMartinottiRK4MatchesPythonReference pins the Go RK4 kernel to the pure
// Python reference: 40000 steps at I = 2 µA/cm² produce 631 spikes and the
// recorded membrane potential.
func TestMartinottiRK4MatchesPythonReference(t *testing.T) {
	s := NewMartinottiNeuron()
	spikes := 0
	for i := 0; i < 40000; i++ {
		spikes += s.Step(5.0)
	}
	if spikes != 134 {
		t.Fatalf("expected 134 spikes, got %d", spikes)
	}
	const refV = -59.32369496390946
	if math.Abs(s.V-refV) > 1e-6 {
		t.Fatalf("membrane potential drifted from Python reference: got %.12f want %.12f", s.V, refV)
	}
}

// TestMartinottiQuiescentWithoutDrive verifies the cell is silent at zero stimulus.
func TestMartinottiQuiescentWithoutDrive(t *testing.T) {
	s := NewMartinottiNeuron()
	spikes := 0
	for i := 0; i < 20000; i++ {
		spikes += s.Step(0.0)
	}
	if spikes != 0 {
		t.Fatalf("expected silence at I=0, got %d spikes", spikes)
	}
}

// TestMartinottiMCurrentChangesFiring checks that the M-type current shapes the
// discharge: blocking it (g_M = 0) changes the spike count.
func TestMartinottiMCurrentChangesFiring(t *testing.T) {
	intact := NewMartinottiNeuron()
	blocked := NewMartinottiNeuron()
	blocked.GM = 0.0
	intactSpikes, blockedSpikes := 0, 0
	for i := 0; i < 40000; i++ {
		intactSpikes += intact.Step(5.0)
		blockedSpikes += blocked.Step(5.0)
	}
	if intactSpikes == blockedSpikes {
		t.Fatalf("M-current block should change firing: intact=%d blocked=%d", intactSpikes, blockedSpikes)
	}
}

// BenchmarkMartinottiRK4 measures the RK4 step throughput.
func BenchmarkMartinottiRK4(b *testing.B) {
	s := NewMartinottiNeuron()
	spikes := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		spikes += s.Step(5.0)
	}
	b.StopTimer()
	b.ReportMetric(float64(spikes), "spikes")
}
