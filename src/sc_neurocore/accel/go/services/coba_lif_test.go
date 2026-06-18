// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for COBA LIF RK4 service

package services

import (
	"math"
	"testing"
)

func TestCOBALIFRK4ConductanceInjection(t *testing.T) {
	state := NewCOBALIFNeuron()
	vCandidate, geCandidate, giCandidate := state.rk4Candidate(state.V, 5.0, 3.0, 0.0)

	spike, err := state.StepWithConductance(0.0, 5.0, 3.0)
	if err != nil {
		t.Fatalf("unexpected step error: %v", err)
	}
	if spike != 0 {
		t.Fatalf("unexpected spike: %d", spike)
	}
	if math.Abs(state.V-vCandidate) > 1.0e-12 || math.Abs(state.GE-geCandidate) > 1.0e-12 || math.Abs(state.GI-giCandidate) > 1.0e-12 {
		t.Fatalf("RK4 candidate mismatch: got (%g,%g,%g), want (%g,%g,%g)", state.V, state.GE, state.GI, vCandidate, geCandidate, giCandidate)
	}
}

func TestCOBALIFInvalidStateDoesNotMutate(t *testing.T) {
	state := NewCOBALIFNeuron()
	state.GE = -1.0
	before := *state
	if _, err := state.Step(0.0); err == nil {
		t.Fatal("expected invalid conductance error")
	}
	if *state != before {
		t.Fatalf("invalid state mutated: got %+v want %+v", *state, before)
	}
}

func TestCOBALIFSuprathresholdResetPreservesConductanceCandidate(t *testing.T) {
	state := NewCOBALIFNeuron()
	state.V = -51.0
	_, geCandidate, _ := state.rk4Candidate(state.V, 5.0, 0.0, 1.0e5)

	spike, err := state.StepWithConductance(1.0e5, 5.0, 0.0)
	if err != nil {
		t.Fatalf("unexpected step error: %v", err)
	}
	if spike != 1 {
		t.Fatalf("expected spike, got %d", spike)
	}
	if state.V != state.VReset {
		t.Fatalf("voltage did not reset: got %g want %g", state.V, state.VReset)
	}
	if math.Abs(state.GE-geCandidate) > 1.0e-12 {
		t.Fatalf("conductance candidate not preserved: got %g want %g", state.GE, geCandidate)
	}
}

func BenchmarkCOBALIFRK4(b *testing.B) {
	const current = 500.0
	state := NewCOBALIFNeuron()
	spikes := 0
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result, err := state.Step(current)
		if err != nil {
			b.Fatalf("invalid RK4 step: %v", err)
		}
		spikes += result
	}
	b.StopTimer()
	b.ReportMetric(float64(spikes), "spikes")
}
