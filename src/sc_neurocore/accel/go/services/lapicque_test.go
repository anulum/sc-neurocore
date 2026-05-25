// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for lapicque

package services

import "testing"

func TestLapicqueStepMatchesEulerCurrentBalance(t *testing.T) {
	state := NewLapicqueNeuron()
	v0 := state.V
	current := 0.5
	expected := v0 + (-(v0-state.VRest)+state.Resistance*current)/state.Tau*state.Dt

	spike, err := state.Step(current)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spike != 0 {
		t.Fatalf("subthreshold input emitted spike: %d", spike)
	}
	if state.V != expected {
		t.Fatalf("voltage mismatch: got %.17g want %.17g", state.V, expected)
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
	state.Tau = 1.0e-308

	if _, err := state.Step(1.0e308); err == nil {
		t.Fatalf("expected overflowing increment to fail")
	}
	if state.V != 0.25 {
		t.Fatalf("overflowing increment mutated voltage to %.17g", state.V)
	}
}
