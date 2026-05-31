// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for morris_lecar

package services

import (
	"math"
	"testing"
)

func closeMorrisLecar(got float64, want float64, tol float64) bool {
	return math.Abs(got-want) <= tol
}

func TestMorrisLecarStepMatchesEulerCurrentBalance(t *testing.T) {
	state := NewMorrisLecarNeuron()
	v0, w0 := state.V, state.W
	current := 50.0
	mInf := 0.5 * (1.0 + math.Tanh((v0-state.V1)/state.V2))
	wInf := 0.5 * (1.0 + math.Tanh((v0-state.V3)/state.V4))
	lam := state.Phi * math.Cosh((v0-state.V3)/(2.0*state.V4))
	iCa := state.GCa * mInf * (v0 - state.ECa)
	iK := state.GK * w0 * (v0 - state.EK)
	iL := state.GL * (v0 - state.EL)
	expectedV := v0 + (-iCa-iK-iL+current)/state.CM*state.Dt
	expectedW := w0 + lam*(wInf-w0)*state.Dt

	state.Step(current)

	if !closeMorrisLecar(state.V, expectedV, 1e-12) {
		t.Fatalf("voltage mismatch: got %.17g want %.17g", state.V, expectedV)
	}
	if !closeMorrisLecar(state.W, expectedW, 1e-12) {
		t.Fatalf("recovery mismatch: got %.17g want %.17g", state.W, expectedW)
	}
}

func TestMorrisLecarRejectsInvalidState(t *testing.T) {
	state := NewMorrisLecarNeuron()
	state.CM = 0.0
	v0, w0 := state.V, state.W

	if got := state.Step(50.0); got != -1 {
		t.Fatalf("invalid state must fail closed, got %d", got)
	}
	if state.V != v0 || state.W != w0 {
		t.Fatalf("invalid state mutated: got (%v, %v)", state.V, state.W)
	}
}
