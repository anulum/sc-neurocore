// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go SC resetting-MAT tests

package services

import (
	"math"
	"testing"
)

func TestSCResettingMATHistoricalAnchor(t *testing.T) {
	state := NewSCResettingMATNeuron()
	events := 0
	for index := 0; index < 256; index++ {
		current := 0.0
		if index >= 32 && index < 128 {
			current = 50.0
		} else if index >= 128 {
			if index%2 == 0 {
				current = 20.0
			} else {
				current = 60.0
			}
		}
		events += state.Step(current)
	}
	if events != 13 || state.V != -70.0 || state.Theta1 != 5.262135955944077 || state.Theta2 != 21.149478444493045 {
		t.Fatalf("historical anchor mismatch: events=%d state=%+v", events, *state)
	}
}

func TestSCResettingMATInvalidStepIsAtomic(t *testing.T) {
	state := NewSCResettingMATNeuron()
	before := *state
	if got := state.Step(math.NaN()); got != -1 || *state != before {
		t.Fatalf("invalid step was not atomic: event=%d state=%+v", got, *state)
	}
}
