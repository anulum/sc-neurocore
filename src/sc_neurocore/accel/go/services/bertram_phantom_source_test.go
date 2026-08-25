// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go source identity tests for Bertram phantom

package services

import "testing"

func TestBertramPhantomFourStateDefaults(t *testing.T) {
	state := NewBertramPhantom()
	if !state.valid() || state.V != -43 || state.N != 0.03 || state.S2 != 0.434 || state.CM != 4524 {
		t.Fatal("invalid Bertram author-code defaults")
	}
}

func TestBertramPhantomDynamicPotassiumGate(t *testing.T) {
	state := NewBertramPhantom()
	if event := state.Step(0); event != 0 {
		t.Fatalf("unexpected event=%d", event)
	}
	expected := [4]float64{-42.96246667898054, 0.030142733666228928, 0.0999512959452674, 0.4339985218163737}
	actual := [4]float64{state.V, state.N, state.S1, state.S2}
	for index := range actual {
		delta := actual[index] - expected[index]
		if delta < -5e-13 || delta > 5e-13 {
			t.Fatalf("state[%d]=%.17g expected %.17g", index, actual[index], expected[index])
		}
	}
}

func BenchmarkBertramPhantomSourceRK4(b *testing.B) {
	state := NewBertramPhantom()
	b.ResetTimer()
	for index := 0; index < b.N; index++ {
		state.Step(0)
	}
}
