// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for lapicque

package services

import (
	"math"
	"testing"
)

func TestLapicqueStepMatchesExactRcFlow(t *testing.T) {
	state := NewLapicqueNeuron()
	v0 := state.V
	current := 0.5
	vInf := state.VRest + state.Resistance*current
	expected := vInf + (v0-vInf)*math.Exp(-state.Dt/state.Tau)

	spike, err := state.Step(current)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spike != 0 {
		t.Fatalf("subthreshold input emitted spike: %d", spike)
	}
	if math.Abs(state.V-expected) > 1e-15 {
		t.Fatalf("voltage mismatch: got %.17g want %.17g", state.V, expected)
	}
}

func TestLapicqueExactFlowDiffersFromEulerWhenDtIsLarge(t *testing.T) {
	state := NewLapicqueNeuron()
	state.V = 0.25
	state.Dt = 5.0
	current := 0.5
	v0 := state.V
	vInf := state.VRest + state.Resistance*current
	euler := v0 + (-(v0-state.VRest)+state.Resistance*current)/state.Tau*state.Dt
	expected := vInf + (v0-vInf)*math.Exp(-state.Dt/state.Tau)

	spike, err := state.Step(current)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spike != 0 {
		t.Fatalf("subthreshold input emitted spike: %d", spike)
	}
	if math.Abs(state.V-expected) > 1e-15 {
		t.Fatalf("voltage mismatch: got %.17g want %.17g", state.V, expected)
	}
	if math.Abs(state.V-euler) < 1e-4 {
		t.Fatalf("exact-flow candidate collapsed to Euler: got %.17g euler %.17g", state.V, euler)
	}
}

func TestLapicqueRejectsInvalidRuntimeStateBeforeMutation(t *testing.T) {
	state := NewLapicqueNeuron()
	state.V = 0.25
	state.Tau = 0.0

	if _, err := state.Step(1.0); err == nil {
		t.Fatalf("expected invalid tau to fail")
	}
	if state.V != 0.25 {
		t.Fatalf("invalid runtime state mutated voltage to %.17g", state.V)
	}
}

func TestLapicqueRejectsOverflowingIncrementBeforeMutation(t *testing.T) {
	state := NewLapicqueNeuron()
	state.V = 0.25
	state.VThreshold = 1.0e308
	state.Resistance = 1.0e308

	if _, err := state.Step(1.0e308); err == nil {
		t.Fatalf("expected overflowing candidate to fail")
	}
	if state.V != 0.25 {
		t.Fatalf("overflowing candidate mutated voltage to %.17g", state.V)
	}
}

func BenchmarkLapicqueExactFlow(b *testing.B) {
	state := NewLapicqueNeuron()
	spikes := 0
	for i := 0; i < b.N; i++ {
		spike, err := state.Step(5.0)
		if err != nil {
			b.Fatalf("unexpected error: %v", err)
		}
		spikes += spike
	}
	if !finiteLapicque(state.V) || spikes < 0 {
		b.Fatalf("invalid final state: v=%v spikes=%d", state.V, spikes)
	}
}
