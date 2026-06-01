// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for GLIF RK4 dynamics

package services

import (
	"math"
	"testing"
)

func TestGLIFRK4ReferencePoint(t *testing.T) {
	s := NewGLIFNeuron()
	s.V = -68.0
	s.Theta = -45.0
	s.IAsc1 = 0.4
	s.IAsc2 = -0.2
	if spike := s.Step(4.0); spike != 0 {
		t.Fatalf("first RK4 step should stay subthreshold, got spike %d", spike)
	}
	if math.Abs(s.V-(-67.7924658728125)) > 1e-12 {
		t.Fatalf("unexpected voltage %.17g", s.V)
	}
	if math.Abs(s.Theta-(-45.049541282631253)) > 1e-12 {
		t.Fatalf("unexpected threshold %.17g", s.Theta)
	}
}

func TestGLIFSpikeResetAddsCandidateThreshold(t *testing.T) {
	s := NewGLIFNeuron()
	s.V = -51.0
	s.Theta = -50.5
	s.DeltaTheta = 2.5
	s.RAsc1 = 1.25
	s.RAsc2 = -0.25
	if spike := s.Step(40.0); spike != 1 {
		t.Fatalf("expected spike, got %d", spike)
	}
	if math.Abs(s.V-(-70.0)) > 1e-12 {
		t.Fatalf("unexpected reset voltage %.17g", s.V)
	}
	if math.Abs(s.Theta-(-47.9930331381625)) > 1e-12 {
		t.Fatalf("unexpected threshold %.17g", s.Theta)
	}
	if math.Abs(s.IAsc1-1.25) > 1e-12 || math.Abs(s.IAsc2-(-0.25)) > 1e-12 {
		t.Fatalf("unexpected current kicks i1=%g i2=%g", s.IAsc1, s.IAsc2)
	}
}

func TestGLIFInvalidInputPreservesState(t *testing.T) {
	s := NewGLIFNeuron()
	s.V = -68.0
	s.IAsc1 = 0.4
	beforeV, beforeTheta, beforeI1, beforeI2 := s.V, s.Theta, s.IAsc1, s.IAsc2
	if spike := s.Step(math.NaN()); spike != 0 {
		t.Fatalf("invalid input should not spike, got %d", spike)
	}
	if s.V != beforeV || s.Theta != beforeTheta || s.IAsc1 != beforeI1 || s.IAsc2 != beforeI2 {
		t.Fatalf("invalid input mutated state: got (%g, %g, %g, %g)", s.V, s.Theta, s.IAsc1, s.IAsc2)
	}
}
