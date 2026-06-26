// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for hay_l5 candidate-first RK4

package services

import (
	"math"
	"testing"
)

func TestHayL5CrossBackendSomaticAnchor(t *testing.T) {
	state := NewHayL5PyramidalNeuron()
	spikes := 0
	for i := 0; i < 20_000; i++ {
		spikes += state.Step(10.0)
	}
	if spikes != 1 {
		t.Fatalf("spikes = %d, want 1", spikes)
	}
	if state.CaA < 0.0 || math.IsNaN(state.CaA) || math.IsInf(state.CaA, 0) {
		t.Fatalf("invalid calcium %g", state.CaA)
	}
}

func TestHayL5CrossBackendDualInputAnchor(t *testing.T) {
	state := NewHayL5PyramidalNeuron()
	spikes := 0
	for i := 0; i < 20_000; i++ {
		spikes += state.Step(5.0, 5.0)
	}
	if spikes != 4 {
		t.Fatalf("spikes = %d, want 4", spikes)
	}
}

func TestHayL5InvalidInputPreservesState(t *testing.T) {
	state := NewHayL5PyramidalNeuron()
	for i := 0; i < 10; i++ {
		_ = state.Step(10.0)
	}
	old := [9]float64{state.VS, state.HNa, state.NK, state.VT, state.MCa, state.HCa, state.MIh, state.VA, state.CaA}
	if state.Step(math.Inf(1)) != 0 {
		t.Fatal("invalid input produced a spike")
	}
	got := [9]float64{state.VS, state.HNa, state.NK, state.VT, state.MCa, state.HCa, state.MIh, state.VA, state.CaA}
	if got != old {
		t.Fatalf("state mutated on invalid input: got %v want %v", got, old)
	}
}

func TestHayL5InvalidConfigurationPreservesState(t *testing.T) {
	state := NewHayL5PyramidalNeuron()
	for i := 0; i < 10; i++ {
		_ = state.Step(10.0)
	}
	old := [9]float64{state.VS, state.HNa, state.NK, state.VT, state.MCa, state.HCa, state.MIh, state.VA, state.CaA}
	state.Dt = 0.0
	if state.Step(10.0) != 0 {
		t.Fatal("invalid configuration produced a spike")
	}
	got := [9]float64{state.VS, state.HNa, state.NK, state.VT, state.MCa, state.HCa, state.MIh, state.VA, state.CaA}
	if got != old {
		t.Fatalf("state mutated on invalid configuration: got %v want %v", got, old)
	}
}

func BenchmarkHayL5RK4(b *testing.B) {
	state := NewHayL5PyramidalNeuron()
	spikes := 0
	b.ReportMetric(0, "spikes")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		spikes += state.Step(10.0)
	}
	b.StopTimer()
	b.ReportMetric(float64(spikes), "spikes")
}
