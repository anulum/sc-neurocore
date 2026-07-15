// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for Wong-Wang Euler/OU dynamics

package services

import (
	"math"
	"testing"
)

func TestWongWangStepUsesPublishedEulerAndOUUpdate(t *testing.T) {
	state := NewWongWangUnit()
	state.S1, state.S2 = 0.24, 0.11
	state.Noise1, state.Noise2 = 0.01, -0.02
	old := *state
	rate1, rate2, err := state.Step(0.17, 0.03, 0.5, -1.0)
	if err != nil {
		t.Fatalf("step failed: %v", err)
	}
	expectedS1 := old.S1 + old.Dt*(-old.S1/old.TauS+(1.0-old.S1)*old.Gamma*rate1)
	expectedS2 := old.S2 + old.Dt*(-old.S2/old.TauS+(1.0-old.S2)*old.Gamma*rate2)
	if state.S1 != expectedS1 || state.S2 != expectedS2 {
		t.Fatalf("Euler mismatch: got %.17g %.17g", state.S1, state.S2)
	}
}

func TestWongWangRejectsInvalidInputAtomically(t *testing.T) {
	state := NewWongWangUnit()
	before := *state
	if _, _, err := state.Step(0.0, 0.0, 0.0, math.NaN()); err == nil {
		t.Fatal("expected invalid Gaussian sample error")
	}
	if state.S1 != before.S1 || state.Noise2 != before.Noise2 {
		t.Fatal("state mutated after rejected input")
	}
}

func TestWongWangResetPreservesParameters(t *testing.T) {
	state := NewWongWangUnit()
	state.TauS, state.TauAMPA, state.Dt = 0.12, 0.003, 0.0002
	state.S1, state.Noise1 = 0.2, 0.4
	state.Reset()
	if state.S1 != 0.1 || state.Noise1 != 0.0 {
		t.Fatal("dynamic state was not restored")
	}
	if state.TauS != 0.12 || state.TauAMPA != 0.003 || state.Dt != 0.0002 {
		t.Fatal("reset changed configured parameters")
	}
}
