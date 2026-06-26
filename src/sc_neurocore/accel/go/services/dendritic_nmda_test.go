// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for dendritic NMDA candidate-first RK4

package services

import (
	"math"
	"testing"
)

func TestDendriticNMDACrossBackendAnchor(t *testing.T) {
	state := NewDendriticNMDANeuron()
	spikes := 0
	for i := 0; i < 20_000; i++ {
		spikes += state.Step(50.0, 0.5)
	}
	if spikes != 253 {
		t.Fatalf("spikes = %d, want 253", spikes)
	}
	if !finite(state.VSoma, state.VDend) {
		t.Fatalf("non-finite final state: soma=%g dend=%g", state.VSoma, state.VDend)
	}
}

func TestDendriticNMDAInvalidInputPreservesState(t *testing.T) {
	state := NewDendriticNMDANeuron()
	for i := 0; i < 10; i++ {
		_ = state.Step(50.0, 0.5)
	}
	old := [2]float64{state.VSoma, state.VDend}
	if state.Step(math.Inf(1), 0.5) != 0 {
		t.Fatal("invalid input produced a spike")
	}
	if got := [2]float64{state.VSoma, state.VDend}; got != old {
		t.Fatalf("state mutated on invalid input: got %v want %v", got, old)
	}
	if state.Step(50.0, -1.0) != 0 {
		t.Fatal("negative glutamate produced a spike")
	}
	if got := [2]float64{state.VSoma, state.VDend}; got != old {
		t.Fatalf("state mutated on negative glutamate: got %v want %v", got, old)
	}
}

func TestDendriticNMDAInvalidConfigurationPreservesState(t *testing.T) {
	state := NewDendriticNMDANeuron()
	for i := 0; i < 10; i++ {
		_ = state.Step(50.0, 0.5)
	}
	old := [2]float64{state.VSoma, state.VDend}
	state.TauDend = 0.0
	if state.Step(50.0, 0.5) != 0 {
		t.Fatal("invalid configuration produced a spike")
	}
	if got := [2]float64{state.VSoma, state.VDend}; got != old {
		t.Fatalf("state mutated on invalid configuration: got %v want %v", got, old)
	}
}

func BenchmarkDendriticNMDARK4(b *testing.B) {
	state := NewDendriticNMDANeuron()
	spikes := 0
	b.ReportMetric(0, "spikes")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		spikes += state.Step(50.0, 0.5)
	}
	b.StopTimer()
	b.ReportMetric(float64(spikes), "spikes")
}
