// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Benda-Herz RK4 service tests

package services

import (
	"math"
	"testing"
)

func bendaHerzReference(s BendaHerzNeuronState, current float64) (float64, float64) {
	rhs := func(a float64) (float64, float64) {
		rate := s.FOnset(current - a)
		return -a/s.TauA + s.DeltaA*rate, rate
	}
	k1, r1 := rhs(s.A)
	k2, r2 := rhs(s.A + 0.5*s.Dt*k1)
	k3, r3 := rhs(s.A + 0.5*s.Dt*k2)
	k4, r4 := rhs(s.A + s.Dt*k3)
	nextA := s.A + (s.Dt/6.0)*(k1+2.0*k2+2.0*k3+k4)
	averageRate := (r1 + 2.0*r2 + 2.0*r3 + r4) / 6.0
	probability := -math.Expm1(-(averageRate * s.Dt / 1000.0))
	return nextA, probability
}

func TestBendaHerzRK4CandidateMatchesReference(t *testing.T) {
	s := NewBendaHerzNeuron()
	s.A = 0.35
	s.Dt = 0.25
	s.Rng = 0.5
	expectedA, expectedP := bendaHerzReference(*s, 12.5)

	nextA, probability, ok := s.RK4Candidate(12.5)

	if !ok {
		t.Fatal("expected valid RK4 candidate")
	}
	if math.Abs(nextA-expectedA) > 1e-12 {
		t.Fatalf("next adaptation mismatch: got %.17g want %.17g", nextA, expectedA)
	}
	if math.Abs(probability-expectedP) > 1e-15 {
		t.Fatalf("probability mismatch: got %.17g want %.17g", probability, expectedP)
	}
}

func TestBendaHerzStepCommitsRK4Candidate(t *testing.T) {
	s := NewBendaHerzNeuron()
	s.A = 0.25
	s.Dt = 0.5
	s.Rng = 0.5
	expectedA, _ := bendaHerzReference(*s, 15.0)

	s.Step(15.0)

	if math.Abs(s.A-expectedA) > 1e-12 {
		t.Fatalf("step committed %.17g, want %.17g", s.A, expectedA)
	}
}

func TestBendaHerzInvalidRuntimeInputPreservesState(t *testing.T) {
	s := NewBendaHerzNeuron()
	s.A = 0.5

	if spike := s.Step(math.NaN()); spike != 0 {
		t.Fatalf("invalid current reported spike %d", spike)
	}
	if s.A != 0.5 {
		t.Fatalf("invalid current mutated adaptation to %.17g", s.A)
	}
}

func TestBendaHerzInvalidCandidatePreservesState(t *testing.T) {
	s := NewBendaHerzNeuron()
	s.A = 0.5
	s.FMax = 1.0e-306
	s.DeltaA = 1.0e308
	s.Dt = 1.0e308

	if spike := s.Step(100.0); spike != 0 {
		t.Fatalf("invalid candidate reported spike %d", spike)
	}
	if s.A != 0.5 {
		t.Fatalf("invalid candidate mutated adaptation to %.17g", s.A)
	}
}

func BenchmarkBendaHerzStep(b *testing.B) {
	s := NewBendaHerzNeuron()
	s.Rng = 0.5
	for i := 0; i < b.N; i++ {
		s.Step(12.5)
	}
}
