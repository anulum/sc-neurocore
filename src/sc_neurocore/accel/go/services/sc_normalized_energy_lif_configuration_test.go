// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Retained normalized energy-LIF Go configuration contracts

package services

import (
	"math"
	"testing"
)

// TestSCNormalizedEnergyLIFRestDomainAndRecovery requires atomic refusal and a valid retry.
func TestSCNormalizedEnergyLIFRestDomainAndRecovery(t *testing.T) {
	for _, rest := range []float64{-201, 101, math.Inf(1)} {
		state := NewSCNormalizedEnergyLIFNeuron()
		state.VRest, state.VThreshold = rest, 102
		beforeV, beforeE := state.V, state.Epsilon
		if state.Valid() || state.Step(0) != -1 || state.V != beforeV || state.Epsilon != beforeE {
			t.Fatal("unsafe rest configuration was accepted or mutated")
		}
		state.VRest = -70
		if !state.Valid() || state.Step(30) != 0 {
			t.Fatal("valid configuration retry failed")
		}
	}
}

// TestSCNormalizedEnergyLIFAllNonfiniteFieldsAreAtomic checks all eleven numeric fields.
func TestSCNormalizedEnergyLIFAllNonfiniteFieldsAreAtomic(t *testing.T) {
	for index := 0; index < 11; index++ {
		for _, bad := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
			state := NewSCNormalizedEnergyLIFNeuron()
			fields := []*float64{&state.V, &state.Epsilon, &state.VRest, &state.VReset, &state.VThreshold, &state.TauM, &state.TauE, &state.Alpha, &state.Epsilon0, &state.Resistance, &state.Dt}
			*fields[index] = bad
			beforeV, beforeE := math.Float64bits(state.V), math.Float64bits(state.Epsilon)
			if state.Valid() || state.Step(30) != -1 || math.Float64bits(state.V) != beforeV || math.Float64bits(state.Epsilon) != beforeE {
				t.Fatalf("nonfinite field %d was accepted or mutated", index)
			}
		}
	}
}
