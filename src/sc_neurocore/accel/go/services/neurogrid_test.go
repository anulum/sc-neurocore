// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for neurogrid candidate-first RK4

package services

import (
	"math"
	"testing"
)

func TestNeuroGridCrossBackendAnchor(t *testing.T) {
	state := NewNeuroGridNeuron()
	spikes := 0
	for i := 0; i < 20_000; i++ {
		spikes += state.Step(100.0)
	}
	if spikes != 94 {
		t.Fatalf("spikes = %d, want 94", spikes)
	}
	if math.IsNaN(state.VS) || math.IsInf(state.VS, 0) || math.IsNaN(state.VD) || math.IsInf(state.VD, 0) {
		t.Fatalf("non-finite final state: vs=%g vd=%g", state.VS, state.VD)
	}
}

func TestNeuroGridInvalidInputPreservesState(t *testing.T) {
	state := NewNeuroGridNeuron()
	for i := 0; i < 10; i++ {
		_ = state.Step(100.0)
	}
	old := [2]float64{state.VS, state.VD}
	if state.Step(math.Inf(1)) != 0 {
		t.Fatal("invalid input produced a spike")
	}
	if got := [2]float64{state.VS, state.VD}; got != old {
		t.Fatalf("state mutated on invalid input: got %v want %v", got, old)
	}
}

func TestNeuroGridInvalidConfigurationPreservesState(t *testing.T) {
	state := NewNeuroGridNeuron()
	for i := 0; i < 10; i++ {
		_ = state.Step(100.0)
	}
	old := [2]float64{state.VS, state.VD}
	state.TauS = 0.0
	if state.Step(100.0) != 0 {
		t.Fatal("invalid configuration produced a spike")
	}
	if got := [2]float64{state.VS, state.VD}; got != old {
		t.Fatalf("state mutated on invalid configuration: got %v want %v", got, old)
	}
}

func BenchmarkNeuroGridRK4(b *testing.B) {
	state := NewNeuroGridNeuron()
	spikes := 0
	b.ReportMetric(0, "spikes")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		spikes += state.Step(100.0)
	}
	b.StopTimer()
	b.ReportMetric(float64(spikes), "spikes")
}
