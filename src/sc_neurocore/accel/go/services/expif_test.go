// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go ExpIF fidelity tests

package services

import (
	"errors"
	"math"
	"testing"
)

func TestExpIFDefaults(t *testing.T) {
	state := NewExpIFNeuron()
	if !state.Valid() {
		t.Fatal("default state must be valid")
	}
	if state.VThreshold != 30.0 || state.VRh != -59.9 || state.DeltaT != 3.48 {
		t.Fatalf("unexpected source defaults: %+v", state)
	}
}

func TestExpIFStepMatchesRK4Reference(t *testing.T) {
	state := NewExpIFNeuron()
	state.V = -62.0
	state.Dt = 0.05
	current := 5.0
	v0 := state.V
	k1 := state.rhs(v0, current)
	k2 := state.rhs(v0+0.5*state.Dt*k1, current)
	k3 := state.rhs(v0+0.5*state.Dt*k2, current)
	k4 := state.rhs(v0+state.Dt*k3, current)
	expected := v0 + (state.Dt/6.0)*(k1+2.0*k2+2.0*k3+k4)

	spike, err := state.Step(current)
	if err != nil || spike != 0 {
		t.Fatalf("unexpected step result: spike=%d err=%v", spike, err)
	}
	if math.Abs(state.V-expected) > 1.0e-12 {
		t.Fatalf("RK4 mismatch: got %.17g want %.17g", state.V, expected)
	}
}

func TestExpIFEventGoldens(t *testing.T) {
	for _, test := range []struct {
		current float64
		spikes  int
	}{{0.0, 0}, {5.0, 0}, {20.0, 2}} {
		state := NewExpIFNeuron()
		total := 0
		for range 1000 {
			spike, err := state.Step(test.current)
			if err != nil {
				t.Fatalf("current %.1f failed: %v", test.current, err)
			}
			total += spike
		}
		if total != test.spikes {
			t.Fatalf("current %.1f: got %d spikes, want %d", test.current, total, test.spikes)
		}
	}
}

func TestExpIFRefractoryHoldAndReset(t *testing.T) {
	state := NewExpIFNeuron()
	state.RefractoryPeriod = 1.7
	for {
		spike, err := state.Step(50.0)
		if err != nil {
			t.Fatal(err)
		}
		if spike == 1 {
			break
		}
	}
	for range 10 {
		spike, err := state.Step(50.0)
		if err != nil || spike != 0 || state.V != state.VReset {
			t.Fatalf("invalid refractory step: spike=%d state=%+v err=%v", spike, state, err)
		}
	}
	if math.Abs(state.RefractoryRemaining-1.5) > 1.0e-12 {
		t.Fatalf("unexpected refractory remainder %.17g", state.RefractoryRemaining)
	}
	state.Reset()
	if state.V != state.VRest || state.RefractoryRemaining != 0.0 {
		t.Fatalf("reset failed: %+v", state)
	}
}

func TestExpIFRejectsInvalidContractsWithoutMutation(t *testing.T) {
	state := NewExpIFNeuron()
	beforeV := state.V
	if _, err := state.Step(math.Inf(1)); !errors.Is(err, ErrExpIFInvalidInput) {
		t.Fatalf("got %v, want invalid input", err)
	}
	if state.V != beforeV {
		t.Fatal("invalid input mutated voltage")
	}

	state.RefractoryRemaining = 1.0
	if _, err := state.Step(0.0); !errors.Is(err, ErrExpIFInvalidState) {
		t.Fatalf("got %v, want invalid state", err)
	}
	if state.V != beforeV || state.RefractoryRemaining != 1.0 {
		t.Fatal("invalid state was partially mutated")
	}
}

func TestExpIFRejectsNonFiniteCandidateWithoutMutation(t *testing.T) {
	state := NewExpIFNeuron()
	state.Dt = 1.0e308
	beforeV := state.V
	if _, err := state.Step(1.0e308); !errors.Is(err, ErrExpIFNonFiniteUpdate) {
		t.Fatalf("got %v, want non-finite update", err)
	}
	if state.V != beforeV {
		t.Fatal("non-finite candidate mutated voltage")
	}
}

func TestSimulateExpIFNeuron(t *testing.T) {
	trace, spikes, err := SimulateExpIFNeuron(1000, 20.0)
	if err != nil || len(trace) != 1000 || spikes != 2 {
		t.Fatalf("unexpected simulation: len=%d spikes=%d err=%v", len(trace), spikes, err)
	}
	if _, _, err := SimulateExpIFNeuron(-1, 0.0); !errors.Is(err, ErrExpIFInvalidState) {
		t.Fatalf("negative step count returned %v", err)
	}
}

func BenchmarkExpIFRK4Step(b *testing.B) {
	state := NewExpIFNeuron()
	b.ResetTimer()
	for range b.N {
		if _, err := state.Step(20.0); err != nil {
			b.Fatal(err)
		}
	}
}
