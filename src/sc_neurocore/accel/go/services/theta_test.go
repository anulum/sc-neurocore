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

func TestThetaSimulateTraceMatchesEventVector(t *testing.T) {
	cases := []struct {
		current float64
		spikes  int
	}{
		{-1.0, 0},
		{-0.5, 0},
		{0.0, 0},
		{0.1, 1},
		{0.333, 2},
		{0.5, 2},
		{1.0, 3},
		{2.0, 5},
		{5.0, 7},
		{20.0, 14},
		{50.0, 23},
	}
	for _, test := range cases {
		trace, spikes, finalTheta, err := SimulateThetaTrace(*NewThetaNeuron(), 1_000, test.current)
		if err != nil {
			t.Fatalf("current=%v: unexpected error: %v", test.current, err)
		}
		if len(trace) != 1_000 || spikes != test.spikes || finalTheta != trace[len(trace)-1] {
			t.Fatalf(
				"current=%v: len=%d spikes=%d final=%.17g",
				test.current,
				len(trace),
				spikes,
				finalTheta,
			)
		}
	}
}

func TestThetaSimulateTracePreservesConfiguredAndEmptyContracts(t *testing.T) {
	initial := ThetaNeuronState{Theta: 0.37, Dt: 0.037}
	trace, spikes, finalTheta, err := SimulateThetaTrace(initial, 400, 2.2)
	if err != nil {
		t.Fatalf("unexpected configured error: %v", err)
	}
	if len(trace) != 400 || spikes != 7 || finalTheta != trace[len(trace)-1] {
		t.Fatalf("unexpected configured result: len=%d spikes=%d final=%.17g", len(trace), spikes, finalTheta)
	}

	empty, emptySpikes, emptyFinal, err := SimulateThetaTrace(initial, 0, 2.2)
	if err != nil {
		t.Fatalf("unexpected empty error: %v", err)
	}
	if len(empty) != 0 || emptySpikes != 0 || emptyFinal != initial.Theta {
		t.Fatalf("unexpected empty result: len=%d spikes=%d final=%.17g", len(empty), emptySpikes, emptyFinal)
	}
}

func TestThetaSimulateTraceRejectsInvalidContracts(t *testing.T) {
	valid := *NewThetaNeuron()
	cases := []struct {
		state   ThetaNeuronState
		steps   int
		current float64
	}{
		{valid, -1, 0.0},
		{ThetaNeuronState{Theta: math.NaN(), Dt: 0.01}, 1, 0.0},
		{ThetaNeuronState{Theta: 0.0, Dt: 0.0}, 1, 0.0},
		{valid, 1, math.Inf(1)},
	}
	for _, test := range cases {
		trace, spikes, finalTheta, err := SimulateThetaTrace(test.state, test.steps, test.current)
		if err != ErrThetaInvalidState {
			t.Fatalf("expected invalid-state error, got %v", err)
		}
		if trace != nil || spikes != 0 || math.Float64bits(finalTheta) != math.Float64bits(test.state.Theta) {
			t.Fatalf("rejected contract emitted state: trace=%v spikes=%d final=%v", trace, spikes, finalTheta)
		}
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
