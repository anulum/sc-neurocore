// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for non_resetting_lif

package services

import (
	"math"
	"testing"
)

func nonResettingLIFExactReference(state float64, steadyState float64, dt float64, tau float64) float64 {
	decay := math.Exp(-dt / tau)
	return decay*state + (1.0-decay)*steadyState
}

func TestNonResettingLIFExactRelaxation(t *testing.T) {
	state := NewNonResettingLIFNeuron()
	state.V = -60.0
	state.Theta = -40.0
	state.Dt = 0.5

	expectedV := nonResettingLIFExactReference(state.V, state.VRest+state.RM*4.0, state.Dt, state.TauM)
	expectedTheta := nonResettingLIFExactReference(state.Theta, state.ThetaRest, state.Dt, state.TauTheta)
	spike, err := state.Step(4.0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spike != 0 {
		t.Fatalf("unexpected spike: %d", spike)
	}
	if math.Abs(state.V-expectedV) > 1e-12 {
		t.Fatalf("V mismatch: got %.17g want %.17g", state.V, expectedV)
	}
	if math.Abs(state.Theta-expectedTheta) > 1e-12 {
		t.Fatalf("theta mismatch: got %.17g want %.17g", state.Theta, expectedTheta)
	}
}

func TestNonResettingLIFLargeTimestepBounded(t *testing.T) {
	state := NewNonResettingLIFNeuron()
	state.V = 1000.0
	state.Theta = 2000.0
	state.Dt = 100.0

	spike, err := state.Step(0.0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spike != 0 {
		t.Fatalf("unexpected spike: %d", spike)
	}
	if state.V < state.VRest || state.V > 1000.0 {
		t.Fatalf("V escaped relaxation envelope: %.17g", state.V)
	}
	if state.Theta < state.ThetaRest || state.Theta > 2000.0 {
		t.Fatalf("theta escaped relaxation envelope: %.17g", state.Theta)
	}
}

func TestNonResettingLIFInvalidUpdatePreservesState(t *testing.T) {
	state := NewNonResettingLIFNeuron()
	state.V = -60.0
	state.Theta = -45.0
	state.RM = 10.0
	beforeV := state.V
	beforeTheta := state.Theta

	if _, err := state.Step(1.0e308); err == nil {
		t.Fatal("expected non-finite exact relaxation error")
	}
	if state.V != beforeV || state.Theta != beforeTheta {
		t.Fatalf("state mutated after invalid update: got %.17g %.17g", state.V, state.Theta)
	}
}
