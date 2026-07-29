// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Aihara source-equation tests

package services

import (
	"math"
	"testing"
)

func TestAiharaSourceStep(t *testing.T) {
	state := NewAiharaMapNeuron()
	expected := 0.7*0.1 - 1.0/(1.0+math.Exp(-10.0)) + 0.3968
	event, err := state.Step(0.0)
	if err != nil || event != 0 || math.Abs(state.Y-expected) > 1e-15 {
		t.Fatalf("unexpected first step: y=%g event=%d err=%v", state.Y, event, err)
	}
}

func TestAiharaLevelEventAndAtomicFailure(t *testing.T) {
	state := &AiharaMapNeuronState{Y: -0.1, K: 0.0, Alpha: 0.01, Bias: 0.2, Epsilon: 0.01}
	for index := 0; index < 2; index++ {
		event, err := state.Step(0.0)
		if err != nil || event != 1 {
			t.Fatalf("level event %d: event=%d err=%v", index, event, err)
		}
	}
	before := state.Y
	if _, err := state.Step(math.NaN()); err == nil || state.Y != before {
		t.Fatal("non-finite input must fail without mutation")
	}
}
