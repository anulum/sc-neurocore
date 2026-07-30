// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

package services

import (
	"math"
	"testing"
)

func TestNonResettingLIFSourceStep(t *testing.T) {
	state := NewNonResettingLIFNeuron()
	state.V = 20.0
	expectedV := 20.0 + state.Dt*(-20.0)/state.TauM
	spike, err := state.Step(0.0)
	if err != nil || spike != 1 {
		t.Fatalf("unexpected event/error: %d %v", spike, err)
	}
	if state.V != expectedV || state.Theta != 37.0 || state.RefractoryRemaining != 2.0 {
		t.Fatalf("source state mismatch: %+v", *state)
	}
	if second, err := state.Step(0.0); err != nil || second != 0 {
		t.Fatalf("refractory gate mismatch: %d %v", second, err)
	}
}

func TestNonResettingLIFInvalidUpdatePreservesState(t *testing.T) {
	state := NewNonResettingLIFNeuron()
	before := *state
	if _, err := state.Step(math.Inf(1)); err == nil {
		t.Fatal("expected invalid-input error")
	}
	if *state != before {
		t.Fatalf("state mutated after invalid input: %+v", *state)
	}
}
