// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for Prescott 2008 RK4 dynamics

package services

import (
	"math"
	"testing"
)

func TestPrescottRK4ReferencePoint(t *testing.T) {
	s := NewPrescottNeuron()
	if spike := s.Step(50.0); spike != 0 {
		t.Fatalf("first RK4 step should stay below threshold, got spike %d", spike)
	}
	if math.Abs(s.V-(-44.498914201492525)) > 1e-12 {
		t.Fatalf("unexpected RK4 voltage %.17g", s.V)
	}
	if math.Abs(s.W-1.4035864179018786e-05) > 1e-17 {
		t.Fatalf("unexpected RK4 recovery %.17g", s.W)
	}
}

func TestPrescottInvalidInputPreservesState(t *testing.T) {
	s := NewPrescottNeuron()
	beforeV, beforeW := s.V, s.W
	if spike := s.Step(math.NaN()); spike != 0 {
		t.Fatalf("invalid input should not spike, got %d", spike)
	}
	if s.V != beforeV || s.W != beforeW {
		t.Fatalf("invalid input mutated state: got (%g, %g)", s.V, s.W)
	}
}

func TestPrescottInvalidCandidatePreservesState(t *testing.T) {
	s := NewPrescottNeuron()
	s.GFast = math.Inf(1)
	beforeV, beforeW := s.V, s.W
	if spike := s.Step(50.0); spike != 0 {
		t.Fatalf("invalid candidate should not spike, got %d", spike)
	}
	if s.V != beforeV || s.W != beforeW {
		t.Fatalf("invalid candidate mutated state: got (%g, %g)", s.V, s.W)
	}
}
