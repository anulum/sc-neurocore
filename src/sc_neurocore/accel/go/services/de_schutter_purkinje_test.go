// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go De Schutter Purkinje RK4 service tests

package services

import (
	"math"
	"testing"
)

func TestDeSchutterPurkinjeCrossBackendSpikeCount(t *testing.T) {
	_, spikes := SimulateDeSchutterPurkinjeNeuron(20000, 500.0)
	if spikes != 1 {
		t.Fatalf("spikes = %d, want 1 (cross-backend RK4 anchor)", spikes)
	}
}

func TestDeSchutterPurkinjeInvalidRuntimeInputPreservesState(t *testing.T) {
	state := NewDeSchutterPurkinjeNeuron()
	for i := 0; i < 10; i++ {
		state.Step(200.0)
	}
	old := *state
	if state.Step(math.Inf(1)) != 0 {
		t.Fatal("invalid current must not report a spike")
	}
	if *state != old {
		t.Fatalf("invalid input mutated state: got %+v, want %+v", *state, old)
	}
}

func TestDeSchutterPurkinjeCalciumNonNegative(t *testing.T) {
	state := NewDeSchutterPurkinjeNeuron()
	for i := 0; i < 20000; i++ {
		state.Step(500.0)
		if state.Ca < 0.0 {
			t.Fatalf("calcium went negative: %g", state.Ca)
		}
	}
}

func BenchmarkDeSchutterPurkinjeRK4(b *testing.B) {
	state := NewDeSchutterPurkinjeNeuron()
	spikes := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		spikes += state.Step(500.0)
	}
	b.StopTimer()
	b.ReportMetric(float64(spikes), "spikes")
}
