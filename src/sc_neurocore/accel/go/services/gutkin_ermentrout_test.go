// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for gutkin_ermentrout

package services

import (
	"math"
	"testing"
)

func closeGutkinErmentrout(got float64, want float64, tol float64) bool {
	return math.Abs(got-want) <= tol
}

func gutkinErmentroutRhsForTest(state *GutkinErmentroutNeuronState, v float64, nGate float64, current float64) (float64, float64) {
	mInf := 1.0 / (1.0 + math.Exp(-(v+20.0)/15.0))
	nInf := 1.0 / (1.0 + math.Exp(-(v+25.0)/5.0))
	iNa := state.GNa * mInf * (v - state.ENa)
	iK := state.GK * nGate * (v - state.EK)
	iL := state.GL * (v - state.EL)
	return -iNa - iK - iL + current, nInf - nGate
}

func TestGutkinErmentroutStepMatchesRK4CurrentBalance(t *testing.T) {
	state := NewGutkinErmentroutNeuron()
	v0, n0 := state.V, state.N
	current := 5.0
	k1V, k1N := gutkinErmentroutRhsForTest(state, v0, n0, current)
	k2V, k2N := gutkinErmentroutRhsForTest(state, v0+0.5*state.Dt*k1V, n0+0.5*state.Dt*k1N, current)
	k3V, k3N := gutkinErmentroutRhsForTest(state, v0+0.5*state.Dt*k2V, n0+0.5*state.Dt*k2N, current)
	k4V, k4N := gutkinErmentroutRhsForTest(state, v0+state.Dt*k3V, n0+state.Dt*k3N, current)
	expectedV := v0 + state.Dt*(k1V+2.0*k2V+2.0*k3V+k4V)/6.0
	expectedN := n0 + state.Dt*(k1N+2.0*k2N+2.0*k3N+k4N)/6.0

	state.Step(current)

	if !closeGutkinErmentrout(state.V, expectedV, 1e-12) {
		t.Fatalf("voltage mismatch: got %.17g want %.17g", state.V, expectedV)
	}
	if !closeGutkinErmentrout(state.N, expectedN, 1e-12) {
		t.Fatalf("gate mismatch: got %.17g want %.17g", state.N, expectedN)
	}
}

func TestGutkinErmentroutRejectsInvalidState(t *testing.T) {
	state := NewGutkinErmentroutNeuron()
	state.N = 1.5
	v0, n0 := state.V, state.N

	if got := state.Step(5.0); got != -1 {
		t.Fatalf("invalid state must fail closed, got %d", got)
	}
	if state.V != v0 || state.N != n0 {
		t.Fatalf("invalid state mutated: got (%v, %v)", state.V, state.N)
	}
}

func TestGutkinErmentroutRejectsInvalidCurrentWithoutMutation(t *testing.T) {
	state := NewGutkinErmentroutNeuron()
	v0, n0 := state.V, state.N

	if got := state.Step(math.NaN()); got != -1 {
		t.Fatalf("invalid current must fail closed, got %d", got)
	}
	if state.V != v0 || state.N != n0 {
		t.Fatalf("invalid current mutated: got (%v, %v)", state.V, state.N)
	}
}

func BenchmarkGutkinErmentroutRK4(b *testing.B) {
	state := NewGutkinErmentroutNeuron()
	spikes := 0
	for i := 0; i < b.N; i++ {
		result := state.Step(5.0)
		if result < 0 {
			b.Fatalf("invalid RK4 step at iteration %d", i)
		}
		spikes += result
	}
	if !finiteGutkinErmentrout(state.V) || !finiteGutkinErmentrout(state.N) {
		b.Fatalf("non-finite final state: (%v, %v)", state.V, state.N)
	}
	b.ReportMetric(float64(spikes), "spikes")
}
