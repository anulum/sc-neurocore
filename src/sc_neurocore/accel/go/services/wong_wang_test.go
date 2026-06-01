// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for Wong-Wang RK4 dynamics

package services

import (
	"math"
	"testing"
)

func TestWongWangStepUsesRK4CoupledState(t *testing.T) {
	state := NewWongWangUnit()
	state.S1 = 0.24
	state.S2 = 0.11
	state.Sigma = 0.0
	state.Dt = 0.02

	r1, r2, err := state.Step(0.17, 0.03, 0.0, 0.0)
	if err != nil {
		t.Fatalf("step failed: %v", err)
	}

	eulerS1 := math.Min(1.0, math.Max(0.0, 0.24+(-0.24/0.1+(1.0-0.24)*0.641*r1)*0.02))
	eulerS2 := math.Min(1.0, math.Max(0.0, 0.11+(-0.11/0.1+(1.0-0.11)*0.641*r2)*0.02))
	if math.Abs(state.S1-eulerS1) <= 1e-5 {
		t.Fatalf("s1 followed forward Euler: got %.17g, euler %.17g", state.S1, eulerS1)
	}
	if math.Abs(state.S2-eulerS2) <= 1e-5 {
		t.Fatalf("s2 followed forward Euler: got %.17g, euler %.17g", state.S2, eulerS2)
	}
}

func TestWongWangRejectsCorruptedRuntimeParameters(t *testing.T) {
	state := NewWongWangUnit()
	state.Dt = 0.0
	beforeS1, beforeS2 := state.S1, state.S2
	if _, _, err := state.Step(0.1, 0.0, 0.0, 0.0); err == nil {
		t.Fatal("expected invalid runtime parameter error")
	}
	if state.S1 != beforeS1 || state.S2 != beforeS2 {
		t.Fatalf("state mutated on invalid parameters: got %.17g %.17g", state.S1, state.S2)
	}
}
