// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service tests for TTypeCaNeuron

package services

import (
	"math"
	"testing"
)

func TestTTypeNominalStep(t *testing.T) {
	state := NewTTypeCaNeuron()
	if spike, err := state.TryStep(5.0); err != nil || spike != 0 {
		t.Fatalf("unexpected step result: spike=%d err=%v", spike, err)
	}
	if math.Abs(state.V-(-63.1681363402518)) > 1e-12 {
		t.Fatalf("unexpected voltage %.17g", state.V)
	}
	if math.Abs(state.S-0.8920254272047233) > 1e-12 {
		t.Fatalf("unexpected s %.17g", state.S)
	}
}

func TestTTypeInvalidDriveIsAtomic(t *testing.T) {
	state := NewTTypeCaNeuron()
	before := *state
	if _, err := state.TryStep(math.NaN()); err == nil {
		t.Fatal("NaN drive must fail")
	}
	if _, err := state.TryStep(math.Inf(1)); err == nil {
		t.Fatal("+Inf drive must fail")
	}
	if _, err := state.TryStep(math.Inf(-1)); err == nil {
		t.Fatal("-Inf drive must fail")
	}
	if *state != before {
		t.Fatal("invalid drive mutated state")
	}
}

func TestTTypeInvalidConfigurationIsAtomic(t *testing.T) {
	state := NewTTypeCaNeuron()
	state.CM = 0.0
	before := *state
	if _, err := state.TryStep(1.0); err == nil {
		t.Fatal("invalid configuration must fail")
	}
	if *state != before {
		t.Fatal("invalid configuration mutated state")
	}
}

func TestTTypeSpikeCollapsesInactivation(t *testing.T) {
	state := NewTTypeCaNeuron()
	fired := 0
	for i := 0; i < 2000 && fired == 0; i++ {
		spike, err := state.TryStep(5.0)
		if err != nil {
			t.Fatalf("unexpected error at step %d: %v", i, err)
		}
		fired += spike
	}
	if fired == 0 {
		t.Fatal("T-type neuron must fire with sustained input")
	}
	if state.S >= 0.9 {
		t.Fatalf("spike must collapse s below its resting value, got %.17g", state.S)
	}
}

func TestTTypeResetPreservesParameters(t *testing.T) {
	state := NewTTypeCaNeuron()
	state.GT, state.V, state.S = 0.5, -30.0, 0.2
	state.Reset()
	if state.V != -65.0 || state.S != 0.9 || state.GT != 0.5 {
		t.Fatalf("unexpected reset state: %+v", state)
	}
}
