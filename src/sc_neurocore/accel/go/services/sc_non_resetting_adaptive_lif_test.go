// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

package services

import (
	"math"
	"testing"
)

func TestSCNonResettingAdaptiveLIFExactRelaxation(t *testing.T) {
	state := NewSCNonResettingAdaptiveLIFNeuron()
	state.V = -60.0
	state.Theta = -40.0
	state.Dt = 0.5
	dv := math.Exp(-state.Dt / state.TauM)
	dt := math.Exp(-state.Dt / state.TauTheta)
	expectedV := dv*state.V + (1.0-dv)*(state.VRest+state.RM*4.0)
	expectedTheta := dt*state.Theta + (1.0-dt)*state.ThetaRest
	spike, err := state.Step(4.0)
	if err != nil || spike != 0 {
		t.Fatalf("unexpected event/error: %d %v", spike, err)
	}
	if math.Abs(state.V-expectedV) > 1e-12 || math.Abs(state.Theta-expectedTheta) > 1e-12 {
		t.Fatalf("exact-relaxation mismatch: %+v", *state)
	}
}

func TestSCNonResettingAdaptiveLIFInvalidInputIsAtomic(t *testing.T) {
	state := NewSCNonResettingAdaptiveLIFNeuron()
	before := *state
	if _, err := state.Step(math.NaN()); err == nil {
		t.Fatal("expected invalid-input error")
	}
	if *state != before {
		t.Fatalf("state mutated after invalid input: %+v", *state)
	}
}
