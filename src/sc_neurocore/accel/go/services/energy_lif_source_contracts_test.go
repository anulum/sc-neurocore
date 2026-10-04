// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Fardet-Levina eLIF configuration contracts

package services

import (
	"math"
	"testing"
)

// TestSourceEnergyLIFConfiguration checks refusal, state preservation and retry.
func TestSourceEnergyLIFConfiguration(t *testing.T) {
	for _, values := range [][3]float64{
		{-62.5, 1, 0}, {-201, 1, .5}, {101, 1, .5},
		{-62.5, 11, .5}, {-62.5, math.MaxFloat64, 2}, {-62.5, 1e-300, 1e-300},
	} {
		state := NewEnergyLIFNeuron()
		state.E0, state.Alpha, state.Epsilon0 = values[0], values[1], values[2]
		before := *state
		if state.Valid() || state.Step(80) != -1 || *state != before {
			t.Fatalf("invalid configuration committed: %v", values)
		}
		state.E0, state.Alpha, state.Epsilon0 = -62.5, 1, .5
		if state.Step(80) != 0 || state.V == before.V {
			t.Fatal("valid retry did not advance")
		}
	}
}

// TestSourceEnergyLIFEnvelope admits inclusive equilibrium bounds.
func TestSourceEnergyLIFEnvelope(t *testing.T) {
	for _, values := range [][3]float64{{-200, 10, .5}, {100, .5, .5}} {
		state := NewEnergyLIFNeuron()
		state.E0, state.Alpha, state.Epsilon0 = values[0], values[1], values[2]
		if !state.Valid() {
			t.Fatalf("valid boundary refused: %v", values)
		}
	}
	state := NewEnergyLIFNeuron()
	before := *state
	if state.Step(math.MaxFloat64) != -1 || *state != before {
		t.Fatal("unsafe candidate changed state")
	}
}
