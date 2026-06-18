// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for MAT RK4 service

package services

import (
	"math"
	"testing"
)

func TestMATRK4CandidateCommit(t *testing.T) {
	state := NewMATNeuron()
	state.Theta1 = 0.5
	state.Theta2 = 1.25
	vCandidate, theta1Candidate, theta2Candidate := state.rk4Candidate(state.V, state.Theta1, state.Theta2, 10.0)

	spike := state.Step(10.0)
	if spike != 0 {
		t.Fatalf("unexpected spike: %d", spike)
	}
	if math.Abs(state.V-vCandidate) > 1.0e-12 ||
		math.Abs(state.Theta1-theta1Candidate) > 1.0e-12 ||
		math.Abs(state.Theta2-theta2Candidate) > 1.0e-12 {
		t.Fatalf(
			"RK4 candidate mismatch: got (%g,%g,%g), want (%g,%g,%g)",
			state.V,
			state.Theta1,
			state.Theta2,
			vCandidate,
			theta1Candidate,
			theta2Candidate,
		)
	}
}

func TestMATInvalidStateDoesNotMutate(t *testing.T) {
	state := NewMATNeuron()
	state.Theta1 = -1.0
	before := *state
	if spike := state.Step(10.0); spike != -1 {
		t.Fatalf("expected invalid-state signal, got %d", spike)
	}
	if *state != before {
		t.Fatalf("invalid state mutated: got %+v want %+v", *state, before)
	}
}

func TestMATSpikeAddsThresholdCandidates(t *testing.T) {
	state := NewMATNeuron()
	_, theta1Candidate, theta2Candidate := state.rk4Candidate(state.V, state.Theta1, state.Theta2, 250.0)

	spike := state.Step(250.0)
	if spike != 1 {
		t.Fatalf("expected spike, got %d", spike)
	}
	if state.V != state.VReset {
		t.Fatalf("voltage did not reset: got %g want %g", state.V, state.VReset)
	}
	if math.Abs(state.Theta1-(theta1Candidate+state.H1)) > 1.0e-12 ||
		math.Abs(state.Theta2-(theta2Candidate+state.H2)) > 1.0e-12 {
		t.Fatalf("threshold candidates not retained after spike: got (%g,%g)", state.Theta1, state.Theta2)
	}
}

func BenchmarkMATRK4(b *testing.B) {
	const current = 50.0
	state := NewMATNeuron()
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
