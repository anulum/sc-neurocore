// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for Av-Ron cardiac RK4 dynamics

package services

import (
	"math"
	"testing"
)

func TestAvRonCardiacRK4ReferencePoint(t *testing.T) {
	s := NewAvRonCardiacNeuron()
	s.V = -55.0
	s.H = 0.55
	s.N = 0.35
	s.S = 0.45
	if spike := s.Step(2.0); spike != 0 {
		t.Fatalf("reference point should stay subthreshold, got %d", spike)
	}
	if math.Abs(s.V-(-50.0840498399381)) > 1e-12 {
		t.Fatalf("unexpected voltage %.17g", s.V)
	}
	if math.Abs(s.H-0.5506609782132562) > 1e-15 || math.Abs(s.N-0.34988677751350306) > 1e-15 || math.Abs(s.S-0.4500091998827305) > 1e-15 {
		t.Fatalf("unexpected gates h=%g n=%g s=%g", s.H, s.N, s.S)
	}
}

func TestAvRonCardiacInvalidStatePreserves(t *testing.T) {
	s := NewAvRonCardiacNeuron()
	s.V = -52.0
	s.H = 0.4
	s.N = 0.2
	s.S = 0.8
	beforeV, beforeH, beforeN, beforeS := s.V, s.H, s.N, s.S
	if spike := s.Step(math.NaN()); spike != 0 {
		t.Fatalf("invalid current should not spike, got %d", spike)
	}
	s.Dt = 0.0
	if spike := s.Step(1.0); spike != 0 {
		t.Fatalf("invalid timestep should not spike, got %d", spike)
	}
	if s.V != beforeV || s.H != beforeH || s.N != beforeN || s.S != beforeS {
		t.Fatalf("invalid path mutated state: got (%g, %g, %g, %g)", s.V, s.H, s.N, s.S)
	}
}
