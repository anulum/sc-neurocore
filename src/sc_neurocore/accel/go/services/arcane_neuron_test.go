// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for arcane_neuron

package services

import (
	"math"
	"testing"
)

func arcaneExactReference(state float64, steadyState float64, dt float64, tau float64) float64 {
	decay := math.Exp(-dt / tau)
	return decay*state + (1.0-decay)*steadyState
}

func arcaneStableSigmoidReference(x float64) float64 {
	if x >= 0.0 {
		z := math.Exp(-x)
		return 1.0 / (1.0 + z)
	}
	z := math.Exp(x)
	return z / (1.0 + z)
}

func TestArcaneNeuronExactRelaxationNoSpike(t *testing.T) {
	state := NewArcaneNeuron()
	state.VFast = 0.4
	state.VWork = 0.2
	state.VDeep = 0.01
	state.Theta = 100.0
	state.Dt = 25.0
	for i := range state.NoveltyHistory {
		state.NoveltyHistory[i] = 0.2
	}

	current := 1.5
	confidence := 0.8
	gateInput := state.WGate[0]*current + state.WGate[1]*state.VFast + state.WGate[2]*state.VWork + state.WGate[3]*confidence
	gate := arcaneStableSigmoidReference(gateInput)
	expectedFast := arcaneExactReference(state.VFast, gate*current, state.Dt, state.TauFast)
	expectedWork := arcaneExactReference(state.VWork, 0.0, state.Dt, state.TauWork)

	spike, err := state.Step(current)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spike != 0 {
		t.Fatalf("unexpected spike: %d", spike)
	}
	if math.Abs(state.VFast-expectedFast) > 1.0e-12 {
		t.Fatalf("VFast mismatch: got %.17g want %.17g", state.VFast, expectedFast)
	}
	if math.Abs(state.VWork-expectedWork) > 1.0e-12 {
		t.Fatalf("VWork mismatch: got %.17g want %.17g", state.VWork, expectedWork)
	}
}

func TestArcaneNeuronInvalidStatePreservesState(t *testing.T) {
	state := NewArcaneNeuron()
	state.VFast = 0.25
	state.VWork = 0.1
	state.VDeep = 0.01
	before := [3]float64{state.VFast, state.VWork, state.VDeep}
	state.TauFast = 0.0

	if _, err := state.Step(0.5); err == nil {
		t.Fatal("expected invalid-state error")
	}
	after := [3]float64{state.VFast, state.VWork, state.VDeep}
	if after != before {
		t.Fatalf("state mutated after invalid runtime state: got %v want %v", after, before)
	}
}
