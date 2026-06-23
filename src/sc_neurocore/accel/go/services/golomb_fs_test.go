// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go RK4 parity + benchmark tests for the Golomb-FS neuron

package services

import (
	"math"
	"testing"
)

// TestGolombFSRK4MatchesPythonReference pins the Go RK4 kernel to the pure Python
// reference: 40000 steps of the fast-spiking cell at I = 5 µA/cm² produce 199
// spikes and the recorded membrane potential.
func TestGolombFSRK4MatchesPythonReference(t *testing.T) {
	s := NewGolombFSNeuron()
	spikes := 0
	for i := 0; i < 40000; i++ {
		spikes += s.Step(5.0)
	}
	if spikes != 199 {
		t.Fatalf("expected 199 spikes, got %d", spikes)
	}
	const refV = -48.25297164103906
	if math.Abs(s.V-refV) > 1e-6 {
		t.Fatalf("membrane potential drifted from Python reference: got %.12f want %.12f", s.V, refV)
	}
}

// TestGolombFSQuiescentWithoutDrive verifies the cell is silent at zero stimulus.
func TestGolombFSQuiescentWithoutDrive(t *testing.T) {
	s := NewGolombFSNeuron()
	spikes := 0
	for i := 0; i < 20000; i++ {
		spikes += s.Step(0.0)
	}
	if spikes != 0 {
		t.Fatalf("expected silence at I=0, got %d spikes", spikes)
	}
}

// TestGolombFSKv3ChangesFiring checks that the Kv3 current shapes the discharge:
// blocking it (g_Kv3 = 0) changes the spike count relative to the intact cell.
func TestGolombFSKv3ChangesFiring(t *testing.T) {
	intact := NewGolombFSNeuron()
	blocked := NewGolombFSNeuron()
	blocked.GKv3 = 0.0
	intactSpikes, blockedSpikes := 0, 0
	for i := 0; i < 40000; i++ {
		intactSpikes += intact.Step(5.0)
		blockedSpikes += blocked.Step(5.0)
	}
	if intactSpikes == blockedSpikes {
		t.Fatalf("Kv3 block should change firing: intact=%d blocked=%d", intactSpikes, blockedSpikes)
	}
}

// BenchmarkGolombFSRK4 measures the RK4 step throughput.
func BenchmarkGolombFSRK4(b *testing.B) {
	s := NewGolombFSNeuron()
	spikes := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		spikes += s.Step(5.0)
	}
	b.StopTimer()
	b.ReportMetric(float64(spikes), "spikes")
}
