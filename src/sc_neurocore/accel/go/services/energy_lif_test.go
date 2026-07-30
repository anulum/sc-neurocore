// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for energy-LIF exact flow

package services

import (
	"math"
	"testing"
)

func TestEnergyLIFExactCandidateCommit(t *testing.T) {
	state := NewSCNormalizedEnergyLIFNeuron()
	state.Epsilon = 0.5
	vCandidate, epsilonCandidate := state.exactCandidate(10.0)

	spike := state.Step(10.0)
	if spike != 0 {
		t.Fatalf("unexpected spike: %d", spike)
	}
	if math.Abs(state.V-vCandidate) > 1.0e-12 || math.Abs(state.Epsilon-epsilonCandidate) > 1.0e-12 {
		t.Fatalf("exact candidate mismatch: got (%g,%g), want (%g,%g)", state.V, state.Epsilon, vCandidate, epsilonCandidate)
	}
}

func TestEnergyLIFInvalidStateDoesNotMutate(t *testing.T) {
	state := NewSCNormalizedEnergyLIFNeuron()
	state.Epsilon = -1.0
	before := *state
	if spike := state.Step(10.0); spike != -1 {
		t.Fatalf("expected invalid-state signal, got %d", spike)
	}
	if *state != before {
		t.Fatalf("invalid state mutated: got %+v want %+v", *state, before)
	}
}

func TestEnergyLIFSpikeUsesEnergyCandidate(t *testing.T) {
	state := NewSCNormalizedEnergyLIFNeuron()
	_, epsilonCandidate := state.exactCandidate(250.0)

	spike := state.Step(250.0)
	if spike != 1 {
		t.Fatalf("expected spike, got %d", spike)
	}
	if state.V != state.VReset {
		t.Fatalf("voltage did not reset: got %g want %g", state.V, state.VReset)
	}
	expected := math.Max(0.0, epsilonCandidate-state.Alpha)
	if math.Abs(state.Epsilon-expected) > 1.0e-12 {
		t.Fatalf("post-spike energy mismatch: got %g want %g", state.Epsilon, expected)
	}
}

func BenchmarkEnergyLIFExactFlow(b *testing.B) {
	const current = 50.0
	state := NewSCNormalizedEnergyLIFNeuron()
	spikes := 0
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result := state.Step(current)
		if result < 0 {
			b.Fatal("invalid exact-flow step")
		}
		spikes += result
	}
	b.StopTimer()
	b.ReportMetric(float64(spikes), "spikes")
}
