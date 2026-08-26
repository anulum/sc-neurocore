// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service tests for SKNeuron

package services

import (
	"math"
	"testing"
)

func TestSKNominalStep(t *testing.T) {
	state := NewSKNeuron()
	if spike, err := state.TryStep(5.0); err != nil || spike != 0 {
		t.Fatalf("unexpected step result: spike=%d err=%v", spike, err)
	}
	if math.Abs(state.V-(-63.18006421307219)) > 1e-12 {
		t.Fatalf("unexpected voltage %.17g", state.V)
	}
	if state.Ca != 0.0 {
		t.Fatalf("unexpected ca %.17g", state.Ca)
	}
}

func TestSKInvalidDriveIsAtomic(t *testing.T) {
	state := NewSKNeuron()
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

func TestSKInvalidConfigurationIsAtomic(t *testing.T) {
	state := NewSKNeuron()
	state.CM = 0.0
	before := *state
	if _, err := state.TryStep(1.0); err == nil {
		t.Fatal("invalid configuration must fail")
	}
	if *state != before {
		t.Fatal("invalid configuration mutated state")
	}
}

func TestSKAdaptationAndCaAccumulation(t *testing.T) {
	state := NewSKNeuron()
	early, late := 0, 0
	for i := 0; i < 2000; i++ {
		spike, err := state.TryStep(5.0)
		if err != nil {
			t.Fatalf("unexpected error at step %d: %v", i, err)
		}
		early += spike
	}
	if state.Ca <= 0.0 {
		t.Fatalf("ca must accumulate with firing, got %.17g", state.Ca)
	}
	for i := 0; i < 2000; i++ {
		spike, err := state.TryStep(5.0)
		if err != nil {
			t.Fatalf("unexpected error at step %d: %v", i, err)
		}
		late += spike
	}
	if early < late {
		t.Fatalf("SK must adapt: early=%d late=%d", early, late)
	}
}

func TestSKResetPreservesParameters(t *testing.T) {
	state := NewSKNeuron()
	state.GSk, state.V, state.Ca = 4.0, -30.0, 0.7
	state.Reset()
	if state.V != -65.0 || state.Ca != 0.0 || state.GSk != 4.0 {
		t.Fatalf("unexpected reset state: %+v", state)
	}
}
