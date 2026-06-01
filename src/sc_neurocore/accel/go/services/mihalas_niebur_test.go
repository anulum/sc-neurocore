// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for Mihalas-Niebur RK4 dynamics

package services

import (
	"math"
	"testing"
)

func TestMihalasNieburRK4ReferencePoint(t *testing.T) {
	s := NewMihalasNieburNeuron()
	if spike := s.Step(0.5); spike != 0 {
		t.Fatalf("first RK4 step should stay subthreshold, got spike %d", spike)
	}
	if math.Abs(s.V-0.04758125) > 1e-12 {
		t.Fatalf("unexpected RK4 voltage %.17g", s.V)
	}
	if math.Abs(s.Theta-1.0) > 1e-15 || s.I1 != 0.0 || s.I2 != 0.0 {
		t.Fatalf("unexpected RK4 state theta=%g i1=%g i2=%g", s.Theta, s.I1, s.I2)
	}
}

func TestMihalasNieburSpikeResetUsesB(t *testing.T) {
	s := NewMihalasNieburNeuron()
	s.V = 0.99
	s.B = 0.5
	s.R1 = 1.25
	s.R2 = -0.25
	if spike := s.Step(2.0); spike != 1 {
		t.Fatalf("expected spike, got %d", spike)
	}
	if math.Abs(s.V-0.5430570625) > 1e-12 {
		t.Fatalf("unexpected reset voltage %.17g", s.V)
	}
	if math.Abs(s.I1-1.25) > 1e-15 || math.Abs(s.I2-(-0.25)) > 1e-15 {
		t.Fatalf("unexpected current kicks i1=%g i2=%g", s.I1, s.I2)
	}
}

func TestMihalasNieburInvalidInputPreservesState(t *testing.T) {
	s := NewMihalasNieburNeuron()
	s.V = 0.2
	s.I1 = 0.3
	beforeV, beforeTheta, beforeI1, beforeI2 := s.V, s.Theta, s.I1, s.I2
	if spike := s.Step(math.NaN()); spike != 0 {
		t.Fatalf("invalid input should not spike, got %d", spike)
	}
	if s.V != beforeV || s.Theta != beforeTheta || s.I1 != beforeI1 || s.I2 != beforeI2 {
		t.Fatalf("invalid input mutated state: got (%g, %g, %g, %g)", s.V, s.Theta, s.I1, s.I2)
	}
}
