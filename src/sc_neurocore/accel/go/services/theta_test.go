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

func TestThetaExactPositiveFlow(t *testing.T) {
	s := NewThetaNeuron()
	s.Theta = 1.0
	s.Dt = 0.2
	rootI := math.Sqrt(2.0)
	expected := wrapTheta(2.0 * math.Atan(rootI*math.Tan(math.Atan(math.Tan(s.Theta/2.0)/rootI)+rootI*s.Dt)))
	spike, err := s.Step(2.0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spike != 0 {
		t.Fatalf("unexpected spike: %d", spike)
	}
	if math.Abs(s.Theta-expected) > 1e-12 {
		t.Fatalf("unexpected theta: %.17g want %.17g", s.Theta, expected)
	}
}

func TestThetaExactFlowReportsWithinStepCrossing(t *testing.T) {
	s := NewThetaNeuron()
	s.Theta = 2.5
	s.Dt = 1.0
	spike, err := s.Step(1.0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spike != 1 {
		t.Fatalf("expected exact crossing spike, got %d", spike)
	}
	if s.Theta < -math.Pi || s.Theta > math.Pi {
		t.Fatalf("theta escaped compact phase: %.17g", s.Theta)
	}
}

func TestThetaStableFixedPointPreserved(t *testing.T) {
	s := NewThetaNeuron()
	s.Theta = -math.Pi / 2.0
	s.Dt = 100.0
	spike, err := s.Step(-1.0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spike != 0 || math.Abs(s.Theta+math.Pi/2.0) > 1e-12 {
		t.Fatalf("fixed point changed: spike=%d theta=%.17g", spike, s.Theta)
	}
}

func TestThetaNonFiniteExactCandidatePreservesState(t *testing.T) {
	s := NewThetaNeuron()
	s.Theta = 0.25
	s.Dt = 1.0e308
	before := s.Theta
	if _, err := s.Step(-1.0e308); err == nil {
		t.Fatal("non-finite exact-flow candidate was accepted")
	}
	if s.Theta != before {
		t.Fatalf("state mutated: before=%v after=%v", before, s.Theta)
	}
}

func BenchmarkThetaExactFlow(b *testing.B) {
	s := NewThetaNeuron()
	spikes := 0
	for i := 0; i < b.N; i++ {
		spike, err := s.Step(0.5)
		if err != nil {
			b.Fatalf("unexpected error: %v", err)
		}
		spikes += spike
	}
	if spikes < 0 {
		b.Fatal("unreachable negative spike count")
	}
}
