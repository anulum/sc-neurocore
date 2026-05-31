// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for larter_breakspear

package services

import (
	"math"
	"testing"
)

func finiteFloat64(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func TestLarterBreakspearStepAdvancesFiniteVoltage(t *testing.T) {
	state := NewLarterBreakspearNeuron()
	initial := state.V

	voltage := state.Step(0.15)

	if !finiteFloat64(voltage) {
		t.Fatalf("expected finite voltage, got %v", voltage)
	}
	if voltage == initial {
		t.Fatalf("expected continuous state to advance from %v", initial)
	}
	if !finiteFloat64(state.W) || !finiteFloat64(state.Z) {
		t.Fatalf("expected recovery variables to remain finite: w=%v z=%v", state.W, state.Z)
	}
}

func TestLarterBreakspearRejectsInvalidState(t *testing.T) {
	state := NewLarterBreakspearNeuron()
	state.Dt = 0
	v0, w0, z0 := state.V, state.W, state.Z

	if got := state.Step(0.0); !math.IsNaN(got) {
		t.Fatalf("expected invalid integration step to return NaN, got %v", got)
	}
	if state.V != v0 || state.W != w0 || state.Z != z0 {
		t.Fatalf("invalid step mutated state: got (%v, %v, %v)", state.V, state.W, state.Z)
	}
}
