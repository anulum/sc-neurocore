// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — retained three-state phantom parity

package services

import "testing"

func TestSCThreeStatePhantomOneStepParity(t *testing.T) {
	state := NewSCThreeStatePhantom()
	if event := state.Step(0); event != 0 {
		t.Fatalf("unexpected event=%d", event)
	}
	expected := [3]float64{-49.81865398074262, 0.10000426804815778, 0.09999950000126304}
	actual := [3]float64{state.V, state.S1, state.S2}
	for index := range actual {
		delta := actual[index] - expected[index]
		if delta < -2e-12 || delta > 2e-12 {
			t.Fatalf("state[%d]=%.17g expected %.17g", index, actual[index], expected[index])
		}
	}
}
