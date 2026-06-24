// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Durstewitz D1 PFC neuron RK4 service tests

package services

import (
	"math"
	"testing"
)

// BenchmarkDurstewitzRK4 exercises the candidate-first RK4 step at the 10 nA
// anchor current so the reported spike metric cross-checks the Python, Rust,
// Julia and Mojo backends bit-for-bit.
func BenchmarkDurstewitzRK4(b *testing.B) {
	s := NewDurstewitzDopamineNeuron()
	spikes := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		spikes += s.Step(10.0)
	}
	b.StopTimer()
	b.ReportMetric(float64(spikes), "spikes")
}

func TestDurstewitzCrossBackendSpikeCount(t *testing.T) {
	// Pins the shared reference the Python/Rust/Julia/Mojo kernels reproduce.
	_, spikes := SimulateDurstewitzDopamineNeuron(200000, 10.0)
	if spikes != 925 {
		t.Fatalf("spikes = %d, want 925 (cross-backend RK4 anchor)", spikes)
	}
}

func TestDurstewitzSpontaneousFiring(t *testing.T) {
	// Dopaminergic tonic activity: the cell fires without external drive.
	_, spikes := SimulateDurstewitzDopamineNeuron(10000, 0.0)
	if spikes < 5 {
		t.Fatalf("spontaneous spikes = %d, want >= 5", spikes)
	}
}

func TestDurstewitzMonotonicFI(t *testing.T) {
	last := -1
	for _, current := range []float64{0.0, 10.0, 30.0, 50.0} {
		_, spikes := SimulateDurstewitzDopamineNeuron(10000, current)
		if spikes < last {
			t.Fatalf("f-I not monotone: current=%g gave %d after %d", current, spikes, last)
		}
		last = spikes
	}
}

func TestDurstewitzStaysFiniteUnderExtremeDrive(t *testing.T) {
	s := NewDurstewitzDopamineNeuron()
	for i := 0; i < 200; i++ {
		s.Step(1e4)
	}
	if math.IsNaN(s.V) || math.IsInf(s.V, 0) {
		t.Fatalf("v became non-finite under extreme drive: %v", s.V)
	}
}
