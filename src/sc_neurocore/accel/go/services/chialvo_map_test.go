// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service tests for the Chialvo map

package services

import (
	"math"
	"testing"
)

func TestChialvoMatchesIndependentSourceStep(t *testing.T) {
	state := &ChialvoMapNeuronState{
		X: 0.2, Y: 0.7, A: 0.89, B: 0.6, C: 0.28, K: 0.04, XThreshold: 1.0,
	}
	expectedX := 0.2*0.2*math.Exp(0.7-0.2) + 0.04 + 0.01
	expectedY := 0.89*0.7 - 0.6*0.2 + 0.28
	spiked, err := state.Step(0.01)
	if err != nil || spiked != 0 {
		t.Fatalf("unexpected step result: spike=%d err=%v", spiked, err)
	}
	if state.X != expectedX || state.Y != expectedY {
		t.Fatalf("source step mismatch: got (%g, %g), want (%g, %g)", state.X, state.Y, expectedX, expectedY)
	}
}

func TestChialvoMatchesPythonGoldenEventCounts(t *testing.T) {
	cases := []struct {
		current float64
		spikes  int64
	}{
		{-0.05, 0}, {0.0, 26}, {0.01, 30}, {0.1, 0}, {1.0, 1},
	}
	for _, test := range cases {
		state := NewChialvoMapNeuron()
		_, spikes, err := state.Simulate(1000, test.current)
		if err != nil {
			t.Fatalf("current=%g: %v", test.current, err)
		}
		if spikes != test.spikes {
			t.Fatalf("current=%g: got %d events, want %d", test.current, spikes, test.spikes)
		}
	}
}

func TestChialvoRejectsInvalidInputWithoutMutation(t *testing.T) {
	state := NewChialvoMapNeuron()
	initialX, initialY := state.X, state.Y
	if _, err := state.Step(math.NaN()); err == nil {
		t.Fatal("expected non-finite current to fail")
	}
	if state.X != initialX || state.Y != initialY {
		t.Fatal("invalid current mutated the state")
	}

	state.X = 1.0e308
	extremeX, extremeY := state.X, state.Y
	if _, err := state.Step(0.0); err == nil {
		t.Fatal("expected overflowing candidate to fail")
	}
	if state.X != extremeX || state.Y != extremeY {
		t.Fatal("invalid candidate mutated the state")
	}
}

func TestChialvoResetPreservesParameters(t *testing.T) {
	state := &ChialvoMapNeuronState{
		X: 2.0, Y: -1.0, A: 0.8, B: 0.4, C: 0.2, K: 0.03, XThreshold: 0.75,
	}
	state.Reset()
	if state.X != 0.0 || state.Y != 0.0 {
		t.Fatalf("reset state mismatch: (%g, %g)", state.X, state.Y)
	}
	if state.A != 0.8 || state.B != 0.4 || state.C != 0.2 || state.K != 0.03 || state.XThreshold != 0.75 {
		t.Fatal("reset changed configuration parameters")
	}
}

func TestChialvoSimulationRejectsNegativeLength(t *testing.T) {
	if _, _, err := SimulateChialvoMapNeuron(-1, 0.0); err == nil {
		t.Fatal("expected negative length to fail")
	}
}
