// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

package services

import (
	"math"
	"testing"
)

func mckeanRK4Reference(s *McKeanNeuronState, current float64) (float64, float64) {
	rhs := func(v, w float64) (float64, float64) {
		return s.f(v) - w + current, s.Epsilon * (v - s.Gamma*w)
	}
	dt := s.Dt
	k1v, k1w := rhs(s.V, s.W)
	k2v, k2w := rhs(s.V+0.5*dt*k1v, s.W+0.5*dt*k1w)
	k3v, k3w := rhs(s.V+0.5*dt*k2v, s.W+0.5*dt*k2w)
	k4v, k4w := rhs(s.V+dt*k3v, s.W+dt*k3w)
	return s.V + dt*(k1v+2*k2v+2*k3v+k4v)/6.0, s.W + dt*(k1w+2*k2w+2*k3w+k4w)/6.0
}

func TestMcKeanRK4CurrentBalance(t *testing.T) {
	s := NewMcKeanNeuron()
	s.V = 0.2
	s.W = -0.1
	expectedV, expectedW := mckeanRK4Reference(s, 0.5)
	spike := s.Step(0.5)
	if spike != 0 {
		t.Fatalf("unexpected spike: %d", spike)
	}
	if math.Abs(s.V-expectedV) > 1e-12 {
		t.Fatalf("unexpected v: %.17g want %.17g", s.V, expectedV)
	}
	if math.Abs(s.W-expectedW) > 1e-12 {
		t.Fatalf("unexpected w: %.17g want %.17g", s.W, expectedW)
	}
}

func TestMcKeanInvalidCurrentPreservesState(t *testing.T) {
	s := NewMcKeanNeuron()
	before := [2]float64{s.V, s.W}
	if spike := s.Step(math.NaN()); spike != 0 {
		t.Fatalf("invalid current produced spike: %d", spike)
	}
	after := [2]float64{s.V, s.W}
	if after != before {
		t.Fatalf("state mutated: before=%v after=%v", before, after)
	}
}

func TestMcKeanOverflowCandidatePreservesState(t *testing.T) {
	s := NewMcKeanNeuron()
	s.V = 1.0e308
	s.W = -1.7e308
	before := [2]float64{s.V, s.W}
	if spike := s.Step(1.7e308); spike != 0 {
		t.Fatalf("overflow candidate produced spike: %d", spike)
	}
	after := [2]float64{s.V, s.W}
	if after != before {
		t.Fatalf("state mutated: before=%v after=%v", before, after)
	}
}
