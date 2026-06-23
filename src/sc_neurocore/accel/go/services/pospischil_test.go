// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go RK4 parity + benchmark tests for the Pospischil neuron

package services

import (
	"math"
	"testing"
)

// TestPospischilRK4MatchesPythonReference pins the Go RK4 kernel to the pure
// Python reference: 40000 steps of the regular-spiking cell at I = 7 µA/cm²
// produce 519 spikes and the recorded membrane potential.
func TestPospischilRK4MatchesPythonReference(t *testing.T) {
	s := NewPospischilNeuron()
	spikes := 0
	for i := 0; i < 40000; i++ {
		spikes += s.Step(7.0)
	}
	if spikes != 519 {
		t.Fatalf("expected 519 spikes, got %d", spikes)
	}
	const refV = -44.747282091050124
	if math.Abs(s.V-refV) > 1e-6 {
		t.Fatalf("membrane potential drifted from Python reference: got %.12f want %.12f", s.V, refV)
	}
}

// TestPospischilQuiescentWithoutDrive verifies the regular-spiking cell is silent
// at zero stimulus.
func TestPospischilQuiescentWithoutDrive(t *testing.T) {
	s := NewPospischilNeuron()
	spikes := 0
	for i := 0; i < 40000; i++ {
		spikes += s.Step(0.0)
	}
	if spikes != 0 {
		t.Fatalf("expected silence at I=0, got %d spikes", spikes)
	}
}

// TestPospischilFastSpikingExceedsRegular checks that removing the M current
// (g_M = 0) raises the firing rate by abolishing spike-frequency adaptation.
func TestPospischilFastSpikingExceedsRegular(t *testing.T) {
	rs := NewPospischilNeuron()
	fs := NewPospischilNeuron()
	fs.GM = 0.0
	rsSpikes, fsSpikes := 0, 0
	for i := 0; i < 40000; i++ {
		rsSpikes += rs.Step(7.0)
		fsSpikes += fs.Step(7.0)
	}
	if fsSpikes <= rsSpikes {
		t.Fatalf("fast-spiking (%d) should exceed regular-spiking (%d)", fsSpikes, rsSpikes)
	}
}

// BenchmarkPospischilRK4 measures the RK4 step throughput for the multi-backend
// comparison benchmark.
func BenchmarkPospischilRK4(b *testing.B) {
	s := NewPospischilNeuron()
	spikes := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		spikes += s.Step(5.0)
	}
	b.StopTimer()
	b.ReportMetric(float64(spikes), "spikes")
}
