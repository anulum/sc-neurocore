// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — AdEx service parity and benchmark tests

package services

import (
	"errors"
	"math"
	"testing"
)

func TestAdExMatchesPythonGoldenSpikeCounts(t *testing.T) {
	for _, tc := range []struct {
		current float64
		spikes  int
	}{{0.0, 0}, {200.0, 4}, {500.0, 12}} {
		state := NewAdExNeuron()
		spikes := 0
		for range 1_000 {
			spike, err := state.Step(tc.current)
			if err != nil {
				t.Fatalf("current=%g: %v", tc.current, err)
			}
			spikes += spike
		}
		if spikes != tc.spikes {
			t.Fatalf("current=%g: got %d spikes, want %d", tc.current, spikes, tc.spikes)
		}
	}
}

func TestAdExOneStepMatchesIndependentEulerReference(t *testing.T) {
	state := NewAdExNeuron()
	state.V = -60.0
	state.W = 3.0
	current := 250.0
	arg := math.Max(-20.0, math.Min(20.0, (state.V-state.VRh)/state.DeltaT))
	expTerm := state.DeltaT * math.Exp(arg)
	dv := (-(state.V-state.VRest)+expTerm)/state.Tau + (-state.W+current)/state.CM
	dw := (state.A*(state.V-state.VRest) - state.W) / state.TauW
	wantV := state.V + dv*state.Dt
	wantW := state.W + dw*state.Dt

	spike, err := state.Step(current)
	if err != nil || spike != 0 {
		t.Fatalf("Step() = (%d, %v), want (0, nil)", spike, err)
	}
	if state.V != wantV || state.W != wantW {
		t.Fatalf("state = (%0.17g, %0.17g), want (%0.17g, %0.17g)", state.V, state.W, wantV, wantW)
	}
}

func TestAdExRejectsInvalidInputWithoutMutation(t *testing.T) {
	state := NewAdExNeuron()
	beforeV, beforeW := state.V, state.W
	spike, err := state.Step(math.Inf(1))
	if spike != 0 || !errors.Is(err, ErrAdExInvalidInput) {
		t.Fatalf("Step(+Inf) = (%d, %v)", spike, err)
	}
	if state.V != beforeV || state.W != beforeW {
		t.Fatal("invalid input mutated state")
	}
}

func TestAdExRejectsInvalidStateAndCandidateWithoutMutation(t *testing.T) {
	invalid := NewAdExNeuron()
	invalid.TauW = 0.0
	beforeV, beforeW := invalid.V, invalid.W
	if _, err := invalid.Step(0.0); !errors.Is(err, ErrAdExInvalidState) {
		t.Fatalf("invalid state error = %v", err)
	}
	if invalid.V != beforeV || invalid.W != beforeW {
		t.Fatal("invalid state mutated dynamic state")
	}

	overflow := NewAdExNeuron()
	overflow.Dt = 1.0e308
	beforeV, beforeW = overflow.V, overflow.W
	if _, err := overflow.Step(1.0e308); !errors.Is(err, ErrAdExNonFiniteUpdate) {
		t.Fatalf("overflow error = %v", err)
	}
	if overflow.V != beforeV || overflow.W != beforeW {
		t.Fatal("non-finite candidate mutated state")
	}
}

func TestAdExResetPreservesParameters(t *testing.T) {
	state := NewAdExNeuron()
	state.VRest = -63.0
	state.Dt = 0.2
	state.A = 0.75
	state.V = -51.0
	state.W = 9.0
	state.Reset()
	if state.V != -63.0 || state.W != 0.0 {
		t.Fatalf("reset state = (%g, %g)", state.V, state.W)
	}
	if state.Dt != 0.2 || state.A != 0.75 {
		t.Fatal("reset mutated parameters")
	}
}

func BenchmarkAdExStep(b *testing.B) {
	state := NewAdExNeuron()
	for b.Loop() {
		if _, err := state.Step(500.0); err != nil {
			b.Fatal(err)
		}
	}
}
