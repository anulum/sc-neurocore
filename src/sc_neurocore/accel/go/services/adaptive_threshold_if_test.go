// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go adaptive-threshold IF exact-relaxation tests

package services

import (
	"math"
	"testing"
)

func adaptiveThresholdIFExactReference(s AdaptiveThresholdIFNeuronState, current float64) (float64, float64) {
	vInf := s.VRest + current
	nextV := vInf + (s.V-vInf)*math.Exp(-s.Dt/s.TauM)
	nextTheta := s.ThetaRest + (s.Theta-s.ThetaRest)*math.Exp(-s.Dt/s.TauTheta)
	return nextV, nextTheta
}

func TestAdaptiveThresholdIFExactRelaxationMatchesReference(t *testing.T) {
	s := NewAdaptiveThresholdIFNeuron()
	s.V = -70.0
	s.Theta = -40.0
	s.Dt = 0.25
	expectedV, expectedTheta := adaptiveThresholdIFExactReference(*s, 12.0)

	spike, err := s.Step(12.0)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spike != 0 {
		t.Fatalf("unexpected spike %d", spike)
	}
	if math.Abs(s.V-expectedV) > 1e-12 || math.Abs(s.Theta-expectedTheta) > 1e-12 {
		t.Fatalf("state mismatch: got (%.17g, %.17g), want (%.17g, %.17g)", s.V, s.Theta, expectedV, expectedTheta)
	}
}

func TestAdaptiveThresholdIFLargeDtRemainsBounded(t *testing.T) {
	s := NewAdaptiveThresholdIFNeuron()
	s.V = -70.0
	s.Theta = -30.0
	s.TauM = 0.04
	s.TauTheta = 0.04
	s.Dt = 1.0

	spike, err := s.Step(0.0)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spike != 0 {
		t.Fatalf("unexpected spike %d", spike)
	}
	if math.Abs(s.V-s.VRest) > 1e-8 || math.Abs(s.Theta-s.ThetaRest) > 1e-8 {
		t.Fatalf("large-step relaxation failed: got (%.17g, %.17g)", s.V, s.Theta)
	}
}

func TestAdaptiveThresholdIFInvalidUpdatePreservesState(t *testing.T) {
	s := NewAdaptiveThresholdIFNeuron()
	s.V = 1.0e308
	beforeV, beforeTheta := s.V, s.Theta

	spike, err := s.Step(-1.0e308)

	if err != ErrAdaptiveThresholdIFNonFiniteUpdate {
		t.Fatalf("expected non-finite update, got spike=%d err=%v", spike, err)
	}
	if s.V != beforeV || s.Theta != beforeTheta {
		t.Fatalf("invalid update mutated state to (%.17g, %.17g)", s.V, s.Theta)
	}
}
