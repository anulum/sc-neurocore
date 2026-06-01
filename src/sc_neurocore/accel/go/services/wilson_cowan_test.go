// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service tests for wilson_cowan

package services

import (
	"math"
	"testing"
)

func TestWilsonCowanStepUsesRK4Reference(t *testing.T) {
	state := NewWilsonCowanUnit()
	state.E = 0.24
	state.I = 0.11
	state.Dt = 0.35

	rate, err := state.Step(3.0)
	if err != nil {
		t.Fatalf("unexpected Wilson-Cowan step error: %v", err)
	}

	eulerE := 0.40111014473980233
	eulerI := 0.10924537850891547
	if math.Abs(rate-0.42143718680097664) > 1.0e-15 {
		t.Fatalf("E RK4 reference mismatch: got %.17g", rate)
	}
	if math.Abs(state.I-0.13798020053932203) > 1.0e-15 {
		t.Fatalf("I RK4 reference mismatch: got %.17g", state.I)
	}
	if math.Abs(state.E-eulerE) <= 1.0e-2 || math.Abs(state.I-eulerI) <= 1.0e-2 {
		t.Fatalf("state did not separate from Euler: E %.17g I %.17g", state.E, state.I)
	}
}

func TestWilsonCowanStepPreservesStateOnInvalidRuntimeParameter(t *testing.T) {
	state := NewWilsonCowanUnit()
	state.TauE = 0.0
	beforeE := state.E
	beforeI := state.I

	if _, err := state.Step(3.0); err == nil {
		t.Fatal("expected invalid runtime parameter error")
	}
	if state.E != beforeE || state.I != beforeI {
		t.Fatalf("state mutated after invalid parameter: E %.17g I %.17g", state.E, state.I)
	}
}
