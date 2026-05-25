// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for escape_rate

package services

import "testing"

func TestEscapeRateStepMatchesMembraneBalanceBelowSaturation(t *testing.T) {
	state := NewEscapeRateNeuron()
	v0 := state.V
	current := 10.0
	expected := v0 + (-(v0-state.VRest)+state.Resistance*current)/state.TauM*state.Dt

	spike, err := state.Step(current)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spike != 0 {
		t.Fatalf("unexpected deterministic saturation spike: %d", spike)
	}
	if state.V != expected {
		t.Fatalf("voltage mismatch: got %.17g want %.17g", state.V, expected)
	}
}

func TestEscapeRateRejectsInvalidRuntimeStateBeforeMutation(t *testing.T) {
	state := NewEscapeRateNeuron()
	state.V = -65.0
	state.DeltaU = 0.0

	if _, err := state.Step(1.0); err == nil {
		t.Fatalf("expected invalid delta_u to fail")
	}
	if state.V != -65.0 {
		t.Fatalf("invalid runtime state mutated voltage to %.17g", state.V)
	}
}

func TestEscapeRateRejectsNonFiniteUpdateBeforeMutation(t *testing.T) {
	state := NewEscapeRateNeuron()
	state.V = -65.0
	state.VThreshold = 1.0e308
	state.TauM = 1.0e-308

	if _, err := state.Step(1.0e308); err == nil {
		t.Fatalf("expected overflowing membrane update to fail")
	}
	if state.V != -65.0 {
		t.Fatalf("overflowing update mutated voltage to %.17g", state.V)
	}
}

func TestEscapeRateRejectsNonFiniteHazardBeforeMutation(t *testing.T) {
	state := NewEscapeRateNeuron()
	state.V = -50.0
	state.Rho0 = 1.0e308
	state.Dt = 1.0e308

	if _, err := state.Step(0.0); err == nil {
		t.Fatalf("expected overflowing escape hazard to fail")
	}
	if state.V != -50.0 {
		t.Fatalf("overflowing hazard mutated voltage to %.17g", state.V)
	}
}
