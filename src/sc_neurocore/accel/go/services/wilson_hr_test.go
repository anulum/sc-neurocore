// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Wilson-HR service tests

package services

import (
	"math"
	"testing"
)

func wilsonHRRK4Reference(s *WilsonHRNeuronState, current float64) (float64, float64) {
	rhs := func(v, r float64) (float64, float64) {
		poly := -(17.81 + 47.71*v + 32.63*v*v) * (v - 0.55)
		syn := -26.0 * r * (v + 0.92)
		return poly + syn + current, (-r + 1.35*v + 1.03) / s.TauR
	}
	v0, r0, dt := s.V, s.R, s.Dt
	k1v, k1r := rhs(v0, r0)
	k2v, k2r := rhs(v0+0.5*dt*k1v, r0+0.5*dt*k1r)
	k3v, k3r := rhs(v0+0.5*dt*k2v, r0+0.5*dt*k2r)
	k4v, k4r := rhs(v0+dt*k3v, r0+dt*k3r)
	return v0 + dt*(k1v+2.0*k2v+2.0*k3v+k4v)/6.0,
		r0 + dt*(k1r+2.0*k2r+2.0*k3r+k4r)/6.0
}

func TestWilsonHRRK4CurrentBalance(t *testing.T) {
	s := NewWilsonHRNeuron()
	s.V = -0.4
	s.R = 0.08
	expectedV, expectedR := wilsonHRRK4Reference(s, 0.3)
	spike, err := s.Step(0.3)
	if err != nil {
		t.Fatalf("Step returned error: %v", err)
	}
	if spike != 0 {
		t.Fatalf("expected no spike, got %d", spike)
	}
	if math.Abs(s.V-expectedV) > 1e-14 || math.Abs(s.R-expectedR) > 1e-14 {
		t.Fatalf("state mismatch: got (%g, %g), want (%g, %g)", s.V, s.R, expectedV, expectedR)
	}
}

func TestWilsonHRInvalidCurrentPreservesState(t *testing.T) {
	s := NewWilsonHRNeuron()
	beforeV, beforeR := s.V, s.R
	if _, err := s.Step(math.NaN()); err == nil {
		t.Fatal("expected invalid current error")
	}
	if s.V != beforeV || s.R != beforeR {
		t.Fatalf("state mutated on invalid current: (%g, %g) -> (%g, %g)", beforeV, beforeR, s.V, s.R)
	}
}

func TestWilsonHROverflowCandidatePreservesState(t *testing.T) {
	s := NewWilsonHRNeuron()
	s.V = 1.0e308
	beforeV, beforeR := s.V, s.R
	if _, err := s.Step(0.3); err == nil {
		t.Fatal("expected candidate error")
	}
	if s.V != beforeV || s.R != beforeR {
		t.Fatalf("state mutated on invalid candidate: (%g, %g) -> (%g, %g)", beforeV, beforeR, s.V, s.R)
	}
}
