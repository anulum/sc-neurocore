// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for hindmarsh_rose

package services

import (
	"math"
	"testing"
)

func closeHindmarshRose(got float64, want float64, tol float64) bool {
	return math.Abs(got-want) <= tol
}

func TestHindmarshRoseStepMatchesEulerCurrentBalance(t *testing.T) {
	state := NewHindmarshRoseNeuron()
	x0, y0, z0 := state.X, state.Y, state.Z
	current := 3.0
	expectedX := x0 + (y0-x0*x0*x0+state.B*x0*x0-z0+current)*state.Dt
	expectedY := y0 + (1.0-5.0*x0*x0-y0)*state.Dt
	expectedZ := z0 + state.R*(state.S*(x0-state.XRest)-z0)*state.Dt

	state.Step(current)

	if !closeHindmarshRose(state.X, expectedX, 1e-12) {
		t.Fatalf("x mismatch: got %.17g want %.17g", state.X, expectedX)
	}
	if !closeHindmarshRose(state.Y, expectedY, 1e-12) {
		t.Fatalf("y mismatch: got %.17g want %.17g", state.Y, expectedY)
	}
	if !closeHindmarshRose(state.Z, expectedZ, 1e-12) {
		t.Fatalf("z mismatch: got %.17g want %.17g", state.Z, expectedZ)
	}
}

func TestHindmarshRoseRejectsInvalidState(t *testing.T) {
	state := NewHindmarshRoseNeuron()
	state.Dt = 0.0

	if got := state.Step(3.0); got != 0 {
		t.Fatalf("invalid state must not emit a spike, got %d", got)
	}
	if !math.IsNaN(state.X) {
		t.Fatalf("invalid state must fail closed with NaN x, got %v", state.X)
	}
}
