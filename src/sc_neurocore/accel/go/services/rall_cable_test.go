// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for Rall cable

package services

import (
	"math"
	"testing"
)

func assertRallClose(t *testing.T, got float64, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("got %.17g, want %.17g", got, want)
	}
}

func TestRallCableStepMatchesImplicitSolve(t *testing.T) {
	state := NewRallCableNeuronWithCompartments(3)
	spike := state.Step(100.0)
	if spike != 0 {
		t.Fatalf("unexpected spike: %d", spike)
	}
	assertRallClose(t, state.V[0], -64.99999695179709)
	assertRallClose(t, state.V[1], -64.99877157422763)
	assertRallClose(t, state.V[2], -64.50371903616434)
}

func TestRallCableInvalidCurrentPreservesState(t *testing.T) {
	state := NewRallCableNeuronWithCompartments(3)
	before := append([]float64(nil), state.V...)
	if state.Step(math.NaN()) != -1 {
		t.Fatal("invalid current must return -1")
	}
	for i := range before {
		assertRallClose(t, state.V[i], before[i])
	}
}

func TestRallCableInvalidStatePreservesState(t *testing.T) {
	state := NewRallCableNeuronWithCompartments(3)
	state.V[1] = math.NaN()
	if state.Step(1.0) != -1 {
		t.Fatal("invalid state must return -1")
	}
	if !math.IsNaN(state.V[1]) {
		t.Fatal("invalid state path mutated the corrupt compartment")
	}
}

func BenchmarkRallCableImplicitSolve(b *testing.B) {
	state := NewRallCableNeuronWithCompartments(5)
	spikes := 0
	for i := 0; i < b.N; i++ {
		spikes += state.Step(500.0)
	}
	b.ReportMetric(float64(spikes), "spikes")
}
