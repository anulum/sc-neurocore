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

func TestSCChaoticMapRetainsTwoStateRecurrence(t *testing.T) {
	s := NewSCChaoticMapNeuron()
	s.X = 0.4
	s.Y = -0.2
	expectedX := 0.7*0.4/(1.0+math.Exp(-2.4)) + 0.2 + 0.1
	expectedY := 0.95*-0.2 + 0.05*0.4
	if _, err := s.Step(0.1); err != nil {
		t.Fatal(err)
	}
	if math.Abs(s.X-expectedX) > 1e-15 || math.Abs(s.Y-expectedY) > 1e-15 {
		t.Fatalf("unexpected state: (%g, %g)", s.X, s.Y)
	}
}

func TestSCChaoticMapRejectsNonFiniteInputAtomically(t *testing.T) {
	s := NewSCChaoticMapNeuron()
	if _, err := s.Step(math.NaN()); err == nil {
		t.Fatal("expected invalid current error")
	}
	if s.X != 0.0 || s.Y != 0.0 {
		t.Fatalf("state changed after rejected input: (%g, %g)", s.X, s.Y)
	}
}
