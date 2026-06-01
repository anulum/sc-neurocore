// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for GIF population dynamics

package services

import (
	"math"
	"testing"
)

func TestGIFPopulationExactSubthresholdReferencePoint(t *testing.T) {
	s := NewGIFPopulationNeuronWithSeed(7)
	s.V = -68.0
	s.Eta = 0.4
	if spike := s.Step(4.0); spike != 0 {
		t.Fatalf("reference point should remain subthreshold, got spike %d", spike)
	}
	if math.Abs(s.V-(-67.8370206677805)) > 1e-12 {
		t.Fatalf("unexpected voltage %.17g", s.V)
	}
	if math.Abs(s.Eta-0.398004991677073) > 1e-15 {
		t.Fatalf("unexpected adaptation %.17g", s.Eta)
	}
}

func TestGIFPopulationForcedSpikeAddsDecayedAdaptation(t *testing.T) {
	s := NewGIFPopulationNeuron()
	s.V = -51.0
	s.Eta = 0.3
	s.Theta = -90.0
	s.Lambda0 = 1.0e9
	if spike := s.Step(0.0); spike != 1 {
		t.Fatalf("expected forced spike, got %d", spike)
	}
	if math.Abs(s.V-s.VReset) > 1e-12 {
		t.Fatalf("unexpected reset voltage %.17g", s.V)
	}
	if math.Abs(s.Eta-5.298503743757805) > 1e-15 {
		t.Fatalf("unexpected adaptation %.17g", s.Eta)
	}
}

func TestGIFPopulationInvalidInputPreservesState(t *testing.T) {
	s := NewGIFPopulationNeuron()
	s.V = -62.0
	s.Eta = 0.75
	beforeV, beforeEta := s.V, s.Eta
	if spike := s.Step(math.NaN()); spike != 0 {
		t.Fatalf("invalid input should not spike, got %d", spike)
	}
	s.TauM = 0.0
	if spike := s.Step(1.0); spike != 0 {
		t.Fatalf("invalid parameter should not spike, got %d", spike)
	}
	if s.V != beforeV || s.Eta != beforeEta {
		t.Fatalf("invalid path mutated state: got (%g, %g)", s.V, s.Eta)
	}
}
