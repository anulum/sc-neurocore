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

func hindmarshRoseDerivative(state *HindmarshRoseNeuronState, x float64, y float64, z float64, current float64) [3]float64 {
	return [3]float64{
		y - x*x*x + state.B*x*x - z + current,
		1.0 - 5.0*x*x - y,
		state.R * (state.S*(x-state.XRest) - z),
	}
}

func TestHindmarshRoseStepMatchesRK4CurrentBalance(t *testing.T) {
	state := NewHindmarshRoseNeuron()
	x0, y0, z0 := state.X, state.Y, state.Z
	current := 3.0
	dt := state.Dt
	k1 := hindmarshRoseDerivative(state, x0, y0, z0, current)
	k2 := hindmarshRoseDerivative(state, x0+0.5*dt*k1[0], y0+0.5*dt*k1[1], z0+0.5*dt*k1[2], current)
	k3 := hindmarshRoseDerivative(state, x0+0.5*dt*k2[0], y0+0.5*dt*k2[1], z0+0.5*dt*k2[2], current)
	k4 := hindmarshRoseDerivative(state, x0+dt*k3[0], y0+dt*k3[1], z0+dt*k3[2], current)
	expectedX := x0 + (dt/6.0)*(k1[0]+2.0*k2[0]+2.0*k3[0]+k4[0])
	expectedY := y0 + (dt/6.0)*(k1[1]+2.0*k2[1]+2.0*k3[1]+k4[1])
	expectedZ := z0 + (dt/6.0)*(k1[2]+2.0*k2[2]+2.0*k3[2]+k4[2])

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
	if state.X != -1.6 || state.Y != -10.0 || state.Z != 2.0 {
		t.Fatalf("invalid state must be preserved, got (%v, %v, %v)", state.X, state.Y, state.Z)
	}
}

func TestHindmarshRoseRejectsOverflowCandidateWithoutMutation(t *testing.T) {
	state := NewHindmarshRoseNeuron()
	state.X = 1.0e103
	beforeX, beforeY, beforeZ := state.X, state.Y, state.Z

	if got := state.Step(3.0); got != 0 {
		t.Fatalf("overflow candidate must not emit a spike, got %d", got)
	}
	if state.X != beforeX || state.Y != beforeY || state.Z != beforeZ {
		t.Fatalf("overflow candidate must preserve state, got (%v, %v, %v)", state.X, state.Y, state.Z)
	}
}
