// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go multicompartment MCN RK4 service tests

package services

import (
	"math"
	"testing"
)

func TestMulticompartmentMCNCrossBackendSpikeCount(t *testing.T) {
	_, spikes := SimulateMulticompartmentMCNNeuron(200000, 3.2)
	if spikes != 49999 {
		t.Fatalf("spikes = %d, want 49999 (cross-backend RK4 anchor)", spikes)
	}
}

func TestMulticompartmentMCNApicalGating(t *testing.T) {
	noApical := NewMulticompartmentMCNNeuron()
	withApical := NewMulticompartmentMCNNeuron()
	spikesNo := 0
	spikesYes := 0
	for i := 0; i < 1000; i++ {
		spikesNo += noApical.StepCompartments(2.5, 0.0, 0.0)
		spikesYes += withApical.StepCompartments(2.5, 5.0, 0.0)
	}
	if spikesYes < spikesNo || spikesYes == 0 {
		t.Fatalf("apical drive should preserve or increase firing: no=%d yes=%d", spikesNo, spikesYes)
	}
}

func TestMulticompartmentMCNInvalidRuntimeInputPreservesState(t *testing.T) {
	state := NewMulticompartmentMCNNeuron()
	for i := 0; i < 5; i++ {
		state.Step(3.0)
	}
	oldU, oldBasal, oldApical := state.U, state.VBasal, state.VApical
	if state.Step(math.Inf(1)) != 0 {
		t.Fatal("invalid current must not report a spike")
	}
	if state.U != oldU || state.VBasal != oldBasal || state.VApical != oldApical {
		t.Fatalf("invalid input mutated state: got (%g,%g,%g), want (%g,%g,%g)",
			state.U, state.VBasal, state.VApical, oldU, oldBasal, oldApical)
	}
}

func BenchmarkMulticompartmentMCNRK4(b *testing.B) {
	state := NewMulticompartmentMCNNeuron()
	spikes := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		spikes += state.Step(3.2)
	}
	b.StopTimer()
	b.ReportMetric(float64(spikes), "spikes")
}
