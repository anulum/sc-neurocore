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

func TestQuadraticIFCurrentBalance(t *testing.T) {
	s := NewQuadraticIFNeuron()
	before := s.V
	spike, err := s.Step(0.5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spike != 0 {
		t.Fatalf("unexpected spike: %d", spike)
	}
	rootI := math.Sqrt(0.5)
	expected := rootI * math.Tan(math.Atan(before/rootI)+rootI*s.Dt)
	if math.Abs(s.V-expected) > 1e-12 {
		t.Fatalf("unexpected v: %.17g want %.17g", s.V, expected)
	}
}

func TestQuadraticIFInvalidCurrentPreservesState(t *testing.T) {
	s := NewQuadraticIFNeuron()
	s.V = -0.25
	before := s.V
	if _, err := s.Step(math.NaN()); err == nil {
		t.Fatal("invalid current was accepted")
	}
	if s.V != before {
		t.Fatalf("state mutated: before=%v after=%v", before, s.V)
	}
}

func TestQuadraticIFNonFiniteIncrementPreservesState(t *testing.T) {
	s := NewQuadraticIFNeuron()
	s.V = -0.25
	before := s.V
	if _, err := s.Step(-1.0e308); err == nil {
		t.Fatal("non-finite exact-flow candidate was accepted")
	}
	if s.V != before {
		t.Fatalf("state mutated: before=%v after=%v", before, s.V)
	}
}

func TestQuadraticIFFixedPointPreserved(t *testing.T) {
	s := NewQuadraticIFNeuron()
	spike, err := s.Step(-1.0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spike != 0 || s.V != -1.0 {
		t.Fatalf("fixed point changed: spike=%d v=%.17g", spike, s.V)
	}
}

func TestQuadraticIFExactFlowResetsOnPeakCrossing(t *testing.T) {
	s := NewQuadraticIFNeuron()
	s.V = 0.95
	s.Dt = 0.5
	spike, err := s.Step(1.0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spike != 1 || s.V != s.VReset {
		t.Fatalf("expected reset spike, got spike=%d v=%.17g", spike, s.V)
	}
}
