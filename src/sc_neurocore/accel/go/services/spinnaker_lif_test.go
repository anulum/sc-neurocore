// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for spinnaker_lif

package services

import (
	"math"
	"testing"
)

func spinnakerLIFReferenceStep(s *SpiNNakerLIFNeuronState, current float64) float64 {
	steady := s.VRest + current + s.IOffset
	return steady + (s.V-steady)*math.Exp(-s.Dt/s.TauM)
}

func TestSpiNNakerLIFStepMatchesExactFlow(t *testing.T) {
	state := NewSpiNNakerLIFNeuron()
	want := spinnakerLIFReferenceStep(state, 10.0)

	if got := state.Step(10.0); got != 0 {
		t.Fatalf("subthreshold exact-flow step spiked: %d", got)
	}
	if math.Abs(state.V-want) > 1e-12 {
		t.Fatalf("voltage mismatch: got %.17g want %.17g", state.V, want)
	}
}

func TestSpiNNakerLIFRefractoryBlocksIntegration(t *testing.T) {
	state := NewSpiNNakerLIFNeuron()
	state.RefracCount = 2.0
	v0 := state.V

	if got := state.Step(100.0); got != 0 {
		t.Fatalf("refractory step must not spike: %d", got)
	}
	if state.V != v0 {
		t.Fatalf("refractory step mutated voltage: got %.17g want %.17g", state.V, v0)
	}
	if state.RefracCount != 1.0 {
		t.Fatalf("refractory countdown mismatch: got %.17g", state.RefracCount)
	}
}

func TestSpiNNakerLIFRejectsInvalidCurrentWithoutMutation(t *testing.T) {
	state := NewSpiNNakerLIFNeuron()
	v0, refrac0 := state.V, state.RefracCount
	if got := state.Step(math.NaN()); got != -1 {
		t.Fatalf("invalid current must fail closed, got %d", got)
	}
	if state.V != v0 || state.RefracCount != refrac0 {
		t.Fatalf("invalid current mutated state: got (%v, %v)", state.V, state.RefracCount)
	}
}

func BenchmarkSpiNNakerLIFExactFlow(b *testing.B) {
	state := NewSpiNNakerLIFNeuron()
	spikes := 0
	for i := 0; i < b.N; i++ {
		result := state.Step(30.0)
		if result < 0 {
			b.Fatalf("invalid exact-flow step at iteration %d", i)
		}
		spikes += result
	}
	b.ReportMetric(float64(spikes), "spikes")
}
