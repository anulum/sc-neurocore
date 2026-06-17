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

func morrisLecarRhsForTest(state *MorrisLecarNeuronState, v float64, w float64, current float64) (float64, float64) {
	mInf := 0.5 * (1.0 + math.Tanh((v-state.V1)/state.V2))
	wInf := 0.5 * (1.0 + math.Tanh((v-state.V3)/state.V4))
	lam := state.Phi * math.Cosh((v-state.V3)/(2.0*state.V4))
	iCa := state.GCa * mInf * (v - state.ECa)
	iK := state.GK * w * (v - state.EK)
	iL := state.GL * (v - state.EL)
	return (-iCa - iK - iL + current) / state.CM, lam * (wInf - w)
}

func TestMorrisLecarStepMatchesRK4CurrentBalance(t *testing.T) {
	state := NewMorrisLecarNeuron()
	v0, w0 := state.V, state.W
	current := 50.0
	k1V, k1W := morrisLecarRhsForTest(state, v0, w0, current)
	k2V, k2W := morrisLecarRhsForTest(state, v0+0.5*state.Dt*k1V, w0+0.5*state.Dt*k1W, current)
	k3V, k3W := morrisLecarRhsForTest(state, v0+0.5*state.Dt*k2V, w0+0.5*state.Dt*k2W, current)
	k4V, k4W := morrisLecarRhsForTest(state, v0+state.Dt*k3V, w0+state.Dt*k3W, current)
	expectedV := v0 + state.Dt*(k1V+2.0*k2V+2.0*k3V+k4V)/6.0
	expectedW := w0 + state.Dt*(k1W+2.0*k2W+2.0*k3W+k4W)/6.0

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

func TestMorrisLecarRejectsInvalidCurrentWithoutMutation(t *testing.T) {
	state := NewMorrisLecarNeuron()
	v0, w0 := state.V, state.W

	if got := state.Step(math.NaN()); got != -1 {
		t.Fatalf("invalid current must fail closed, got %d", got)
	}
	if state.V != v0 || state.W != w0 {
		t.Fatalf("invalid current mutated: got (%v, %v)", state.V, state.W)
	}
}

func TestMorrisLecarRejectsOverflowCandidateWithoutMutation(t *testing.T) {
	state := NewMorrisLecarNeuron()
	state.V = 1.0e6
	state.W = 0.25
	v0, w0 := state.V, state.W

	if got := state.Step(0.0); got != -1 {
		t.Fatalf("overflow candidate must fail closed, got %d", got)
	}
	if state.V != v0 || state.W != w0 {
		t.Fatalf("overflow candidate mutated: got (%v, %v)", state.V, state.W)
	}
}

func BenchmarkMorrisLecarRK4(b *testing.B) {
	state := NewMorrisLecarNeuron()
	spikes := 0
	for i := 0; i < b.N; i++ {
		spikes += state.Step(100.0)
	}
	if !finiteMorrisLecar(state.V) || !finiteMorrisLecar(state.W) {
		b.Fatalf("non-finite final state: (%v, %v)", state.V, state.W)
	}
	b.ReportMetric(float64(spikes), "spikes")
}
