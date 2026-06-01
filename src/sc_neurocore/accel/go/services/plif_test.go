// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for plif

package services

import "testing"

func TestParametricLIFCandidateOverflowPreservesState(t *testing.T) {
	state := NewParametricLIFNeuron()
	state.V = 1.0e308
	state.A = 1000.0
	state.Threshold = 1.7e308
	before := state.V

	spike := state.Step(1.0e308)
	if spike != 0 {
		t.Fatalf("unexpected spike for invalid candidate: %d", spike)
	}
	if state.V != before {
		t.Fatalf("state mutated after invalid candidate: got %.17g want %.17g", state.V, before)
	}
}

func TestParametricLIFInvalidRuntimeStatePreservesState(t *testing.T) {
	state := NewParametricLIFNeuron()
	state.V = 0.25
	before := state.V
	state.Threshold = 0.0

	spike := state.Step(0.1)
	if spike != 0 {
		t.Fatalf("unexpected spike for invalid runtime state: %d", spike)
	}
	if state.V != before {
		t.Fatalf("state mutated after invalid runtime state: got %.17g want %.17g", state.V, before)
	}
}
