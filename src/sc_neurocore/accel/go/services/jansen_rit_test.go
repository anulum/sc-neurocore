// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service tests for Jansen–Rit

package services

import (
	"math"
	"testing"
)

func TestJansenRitStepUsesEquationSixConnectivity(t *testing.T) {
	state := NewJansenRitUnit()
	state.Y0 = 0.1
	oldY4 := state.Y4
	se := state.sigmoid(state.C * state.Y0)
	want := oldY4 + state.Dt*(state.AExc*state.ARate*(220.0+0.8*state.C*se)-2.0*state.ARate*oldY4)
	if _, err := state.Step(220.0); err != nil {
		t.Fatal(err)
	}
	if state.Y4 != want {
		t.Fatalf("Y4=%0.17g, want %0.17g", state.Y4, want)
	}
}

func TestJansenRitRejectsNonFiniteInputWithoutMutation(t *testing.T) {
	state := NewJansenRitUnit()
	before := *state
	if _, err := state.Step(math.NaN()); err == nil {
		t.Fatal("expected non-finite drive rejection")
	}
	if *state != before {
		t.Fatal("rejected step mutated state")
	}
}
