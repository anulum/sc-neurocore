// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Hill-Tononi thalamocortical neuron RK4 service tests

package services

import (
	"math"
	"testing"
)

// BenchmarkHillTononiRK4 exercises the candidate-first RK4 step at the 10 nA
// anchor current so the reported spike metric cross-checks the Python, Rust,
// Julia and Mojo backends bit-for-bit.
func BenchmarkHillTononiRK4(b *testing.B) {
	s := NewHillTononiNeuron()
	spikes := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		spikes += s.Step(10.0)
	}
	b.StopTimer()
	b.ReportMetric(float64(spikes), "spikes")
}

func TestHillTononiCrossBackendSpikeCount(t *testing.T) {
	// Pins the shared reference the Python/Rust/Julia/Mojo kernels reproduce.
	_, spikes := SimulateHillTononiNeuron(200000, 10.0)
	if spikes != 694 {
		t.Fatalf("spikes = %d, want 694 (cross-backend RK4 anchor)", spikes)
	}
}

func TestHillTononiSpontaneousFiring(t *testing.T) {
	// Thalamocortical tonic firing at rest current.
	_, spikes := SimulateHillTononiNeuron(10000, 0.0)
	if spikes < 5 {
		t.Fatalf("spontaneous spikes = %d, want >= 5", spikes)
	}
}

func TestHillTononiSodiumStaysNonNegative(t *testing.T) {
	s := NewHillTononiNeuron()
	for i := 0; i < 50000; i++ {
		s.Step(10.0)
		if s.NaI < 0.0 {
			t.Fatalf("na_i went negative: %v", s.NaI)
		}
	}
}

func TestHillTononiStaysFiniteUnderExtremeDrive(t *testing.T) {
	s := NewHillTononiNeuron()
	for i := 0; i < 200; i++ {
		s.Step(1e4)
	}
	if math.IsNaN(s.V) || math.IsInf(s.V, 0) {
		t.Fatalf("v became non-finite under extreme drive: %v", s.V)
	}
}
