// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service tests for the MPR mean field

package services

import (
	"math"
	"testing"
)

func TestErmentroutKopellPopulationStepMatchesEquation12(t *testing.T) {
	state := &ErmentroutKopellPopulationState{
		R: 0.2, V: -1.5, Tau: 2.0, Delta: 0.7, EtaBar: -3.0, J: 12.0, Dt: 0.005,
	}
	expectedR := 0.2 + 0.005*(0.7/(math.Pi*4.0)+2.0*0.2*-1.5/2.0)
	expectedV := -1.5 + 0.005*((-1.5)*(-1.5)-3.0+1.25+12.0*2.0*0.2-math.Pow(math.Pi*2.0*0.2, 2))/2.0
	if _, err := state.Step(1.25); err != nil {
		t.Fatal(err)
	}
	if math.Abs(state.R-expectedR) > 1e-15 || math.Abs(state.V-expectedV) > 1e-15 {
		t.Fatalf("unexpected state: got (%g,%g), want (%g,%g)", state.R, state.V, expectedR, expectedV)
	}
}

func TestErmentroutKopellPopulationInvalidInputIsAtomic(t *testing.T) {
	state := NewErmentroutKopellPopulation()
	beforeR, beforeV := state.R, state.V
	if _, err := state.Step(math.NaN()); err == nil {
		t.Fatal("expected invalid input error")
	}
	if state.R != beforeR || state.V != beforeV {
		t.Fatal("invalid input mutated state")
	}
}

func BenchmarkErmentroutKopellPopulationStep(b *testing.B) {
	state := NewErmentroutKopellPopulation()
	for index := 0; index < b.N; index++ {
		if _, err := state.Step(1.5); err != nil {
			b.Fatal(err)
		}
	}
}
