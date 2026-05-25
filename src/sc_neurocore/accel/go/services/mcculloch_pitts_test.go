// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for mcculloch_pitts

package services

import (
	"math"
	"testing"
)

func TestMcCullochPittsStepMatchesHeavisideBoundary(t *testing.T) {
	state := NewMcCullochPittsNeuron()
	state.Theta = 2.0

	below, err := state.Step(1.999999999999999)
	if err != nil {
		t.Fatalf("unexpected error below threshold: %v", err)
	}
	at, err := state.Step(2.0)
	if err != nil {
		t.Fatalf("unexpected error at threshold: %v", err)
	}

	if below != 0 || at != 1 {
		t.Fatalf("expected Heaviside boundary 0 then 1, got %d then %d", below, at)
	}
}

func TestMcCullochPittsRejectsInvalidRuntimeThreshold(t *testing.T) {
	state := NewMcCullochPittsNeuron()
	state.Theta = math.NaN()

	if _, err := state.Step(1.0); err == nil {
		t.Fatalf("expected invalid threshold to fail")
	}
}

func TestMcCullochPittsRejectsNonFiniteInput(t *testing.T) {
	state := NewMcCullochPittsNeuron()

	if _, err := state.Step(math.Inf(1)); err == nil {
		t.Fatalf("expected non-finite weighted input to fail")
	}
}
