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
	if spikes != 266 {
		t.Fatalf("spikes = %d, want 266", spikes)
	}
	if !finite(state.VSoma, state.VDend) {
		t.Fatalf("non-finite final state: soma=%g dend=%g", state.VSoma, state.VDend)
	}
}

// TestDendriticNMDACoincidenceDetection holds the model's defining behaviour: neither a
// subthreshold soma current nor glutamate alone fires, both together do, and glutamate
// adds spikes at a current that fires alone.
func TestDendriticNMDACoincidenceDetection(t *testing.T) {
	run := func(iSoma, glutamate float64) (int, float64) {
		state := NewDendriticNMDANeuron()
		spikes := 0
		for i := 0; i < 20_000; i++ {
			spikes += state.Step(iSoma, glutamate)
		}
		return spikes, state.VDend
	}
	if spikes, _ := run(20.0, 0.0); spikes != 0 {
		t.Fatalf("soma current 20 alone fired %d times, want 0", spikes)
	}
	spikes, vDend := run(0.0, 1.0)
	if spikes != 0 {
		t.Fatalf("glutamate alone fired %d times, want 0", spikes)
	}
	if vDend <= -60.0 || vDend >= -59.0 {
		t.Fatalf("balance point of the dendrite must lie in (-60, -59): vDend=%g", vDend)
	}
	if both, _ := run(20.0, 1.0); both < 50 || both > 70 {
		t.Fatalf("soma current 20 with glutamate fired %d times, want 50 to 70", both)
	}
	somaOnly, _ := run(40.0, 0.0)
	withGlutamate, _ := run(40.0, 1.0)
	if somaOnly < 180 || somaOnly > 200 {
		t.Fatalf("soma current 40 alone fired %d times, want 180 to 200", somaOnly)
	}
	if withGlutamate < somaOnly+10 || withGlutamate > somaOnly+30 {
		t.Fatalf("glutamate must add 10 to 30 spikes: with=%d somaOnly=%d", withGlutamate, somaOnly)
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
