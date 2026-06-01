// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for leaky_compete_fire

package services

import (
	"math"
	"testing"
)

func lcfExactReference(voltage float64, current float64, tau float64, dt float64) float64 {
	return current + (voltage-current)*math.Exp(-dt/tau)
}

func TestLeakyCompeteFireExactRelaxation(t *testing.T) {
	state := NewLeakyCompeteFireNeuron()
	state.V = []float64{0.2, 0.4, 0.1}
	state.NUnits = 3
	state.Tau = 7.0
	state.VThreshold = 100.0
	state.Dt = 2.5
	currents := []float64{1.0, 0.5, 0.0}
	expected := []float64{
		lcfExactReference(state.V[0], currents[0], state.Tau, state.Dt),
		lcfExactReference(state.V[1], currents[1], state.Tau, state.Dt),
		lcfExactReference(state.V[2], currents[2], state.Tau, state.Dt),
	}

	spikes, err := state.Step(currents)
	if err != nil {
		t.Fatalf("unexpected step error: %v", err)
	}
	for i, spike := range spikes {
		if spike != 0 {
			t.Fatalf("unexpected spike at %d: %d", i, spike)
		}
		if math.Abs(state.V[i]-expected[i]) > 1.0e-12 {
			t.Fatalf("V[%d] mismatch: got %.17g want %.17g", i, state.V[i], expected[i])
		}
	}
}

func TestLeakyCompeteFireInvalidStatePreservesState(t *testing.T) {
	state := NewLeakyCompeteFireNeuron()
	state.V = []float64{0.2, 0.4}
	state.NUnits = 2
	before := append([]float64(nil), state.V...)
	state.Tau = 0.0

	if _, err := state.Step([]float64{1.0, 0.5}); err == nil {
		t.Fatal("expected invalid-state error")
	}
	if state.V[0] != before[0] || state.V[1] != before[1] {
		t.Fatalf("state mutated after invalid runtime state: got %v want %v", state.V, before)
	}
}
