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

func TestMcKeanCurrentBalance(t *testing.T) {
	s := NewMcKeanNeuron()
	spike := s.Step(0.5)
	if spike != 0 {
		t.Fatalf("unexpected spike: %d", spike)
	}
	if math.Abs(s.V-0.05) > 1e-12 {
		t.Fatalf("unexpected v: %.17g", s.V)
	}
	if math.Abs(s.W) > 1e-12 {
		t.Fatalf("unexpected w: %.17g", s.W)
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
