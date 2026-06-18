// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for SFA RK4 service

package services

import (
	"math"
	"testing"
)

func TestSFARK4CandidateCommit(t *testing.T) {
	state := NewSFANeuron()
	state.GSfa = 0.25
	vCandidate, gCandidate := state.rk4Candidate(state.V, state.GSfa, 10.0)

	spike := state.Step(10.0)
	if spike != 0 {
		t.Fatalf("unexpected spike: %d", spike)
	}
	if math.Abs(state.V-vCandidate) > 1.0e-12 || math.Abs(state.GSfa-gCandidate) > 1.0e-12 {
		t.Fatalf("RK4 candidate mismatch: got (%g,%g), want (%g,%g)", state.V, state.GSfa, vCandidate, gCandidate)
	}
}

func TestSFAInvalidStateDoesNotMutate(t *testing.T) {
	state := NewSFANeuron()
	state.GSfa = -1.0
	before := *state
	if spike := state.Step(10.0); spike != -1 {
		t.Fatalf("expected invalid-state signal, got %d", spike)
	}
	if *state != before {
		t.Fatalf("invalid state mutated: got %+v want %+v", *state, before)
	}
}

func TestSFASpikeAddsAdaptationCandidate(t *testing.T) {
	state := NewSFANeuron()
	_, gCandidate := state.rk4Candidate(state.V, state.GSfa, 250.0)

	spike := state.Step(250.0)
	if spike != 1 {
		t.Fatalf("expected spike, got %d", spike)
	}
	if state.V != state.VReset {
		t.Fatalf("voltage did not reset: got %g want %g", state.V, state.VReset)
	}
	if math.Abs(state.GSfa-(gCandidate+state.DeltaG)) > 1.0e-12 {
		t.Fatalf("adaptation candidate mismatch: got %g want %g", state.GSfa, gCandidate+state.DeltaG)
	}
}

func BenchmarkSFARK4(b *testing.B) {
	const current = 50.0
	state := NewSFANeuron()
	spikes := 0
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result := state.Step(current)
		if result < 0 {
			b.Fatal("invalid RK4 step")
		}
		spikes += result
	}
	b.StopTimer()
	b.ReportMetric(float64(spikes), "spikes")
}
