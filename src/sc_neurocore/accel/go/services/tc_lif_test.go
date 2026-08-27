// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service tests for TwoCompartmentLIFNeuron

package services

import (
	"math"
	"testing"
)

func TestTCLIFMatchesIndependentPaperOracle(t *testing.T) {
	beta1, beta2, gamma, vTh := -0.5, 0.5, 0.5, 1.0
	uD, uS, sPrev := 0.0, 0.0, 0.0
	state := NewTwoCompartmentLIFNeuron()
	for index := 0; index < 200; index++ {
		input := 0.3 + 0.2*(float64(index%7)-3.0)
		uD = uD + beta1*uS + input - gamma*sPrev
		uS = uS + beta2*uD - vTh*sPrev
		spike := 0.0
		if uS >= vTh {
			spike = 1.0
		}
		sPrev = spike
		got, err := state.TryStep(input)
		if err != nil {
			t.Fatalf("unexpected error at step %d: %v", index, err)
		}
		if float64(got) != spike || state.UD != uD || state.US != uS {
			t.Fatalf("oracle divergence at step %d", index)
		}
	}
}

func TestTCLIFInvalidDriveIsAtomic(t *testing.T) {
	state := NewTwoCompartmentLIFNeuron()
	before := *state
	if _, err := state.TryStep(math.NaN()); err == nil {
		t.Fatal("NaN drive must fail")
	}
	if _, err := state.TryStep(math.Inf(1)); err == nil {
		t.Fatal("+Inf drive must fail")
	}
	if *state != before {
		t.Fatal("invalid drive mutated state")
	}
}

func TestTCLIFBetaSignsEnforced(t *testing.T) {
	state := NewTwoCompartmentLIFNeuron()
	state.Beta1 = 0.5
	if _, err := state.TryStep(0.0); err == nil {
		t.Fatal("positive beta1 must fail")
	}
	state = NewTwoCompartmentLIFNeuron()
	state.Beta2 = -0.5
	if _, err := state.TryStep(0.0); err == nil {
		t.Fatal("negative beta2 must fail")
	}
}

func TestTCLIFResetPreservesParameters(t *testing.T) {
	state := NewTwoCompartmentLIFNeuron()
	state.Gamma, state.US = 0.7, 0.4
	state.Reset()
	if state.US != 0.0 || state.Gamma != 0.7 {
		t.Fatalf("unexpected reset state: %+v", state)
	}
}
