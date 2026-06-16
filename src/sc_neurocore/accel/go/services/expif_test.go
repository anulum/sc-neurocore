// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go ExpIF RK4 tests

package services

import (
	"math"
	"testing"
)

func expIFRK4Reference(s ExpIFNeuronState, current float64) float64 {
	rhs := func(v float64) float64 {
		arg := (v - s.VRh) / s.DeltaT
		if arg < -20.0 {
			arg = -20.0
		} else if arg > 20.0 {
			arg = 20.0
		}
		return (-(v - s.VRest) + s.DeltaT*math.Exp(arg) + current) / s.Tau
	}
	k1 := rhs(s.V)
	k2 := rhs(s.V + 0.5*s.Dt*k1)
	k3 := rhs(s.V + 0.5*s.Dt*k2)
	k4 := rhs(s.V + s.Dt*k3)
	return s.V + s.Dt*(k1+2.0*k2+2.0*k3+k4)/6.0
}

func TestExpIFStepMatchesRK4Reference(t *testing.T) {
	s := NewExpIFNeuron()
	s.V = -60.0
	current := 12.0
	expected := expIFRK4Reference(*s, current)

	spike, err := s.Step(current)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spike != 0 {
		t.Fatalf("unexpected spike: %d", spike)
	}
	if math.Abs(s.V-expected) > 1e-12 {
		t.Fatalf("RK4 mismatch: got %.17g want %.17g", s.V, expected)
	}
}

func TestExpIFRejectsInvalidUpdateBeforeMutation(t *testing.T) {
	s := NewExpIFNeuron()
	s.V = -60.0
	s.Dt = 1.0e308
	before := s.V

	spike, err := s.Step(1.0e308)
	if err != ErrExpIFNonFiniteUpdate {
		t.Fatalf("expected non-finite update, got spike=%d err=%v", spike, err)
	}
	if s.V != before {
		t.Fatalf("invalid update mutated voltage to %.17g", s.V)
	}
}

func BenchmarkExpIFRK4Step(b *testing.B) {
	s := NewExpIFNeuron()
	for i := 0; i < b.N; i++ {
		_, err := s.Step(20.0)
		if err != nil {
			b.Fatalf("unexpected error: %v", err)
		}
	}
}
