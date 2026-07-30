// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go source/SC energy-LIF parity tests

package services

import (
	"math"
	"testing"
)

func TestEnergyLIFSourceRK4AndAtomicFailure(t *testing.T) {
	state := NewEnergyLIFNeuron()
	v, epsilon := state.rk4Candidate(80)
	if event := state.Step(80); event != 0 || state.V != v || state.Epsilon != epsilon {
		t.Fatalf("source transition mismatch: event=%d state=(%g,%g)", event, state.V, state.Epsilon)
	}
	before := *state
	if state.Step(mathNaN()) != -1 || *state != before {
		t.Fatal("invalid source input was not atomic")
	}
}

func TestSCNormalizedEnergyLIFFrozenEventCount(t *testing.T) {
	state := NewSCNormalizedEnergyLIFNeuron()
	currents := [...]float64{30, 0, 50, 10}
	events := 0
	for i := 0; i < 256; i++ {
		events += state.Step(currents[i%len(currents)])
	}
	if events != 3 {
		t.Fatalf("retained event count=%d, want 3", events)
	}
}

func mathNaN() float64 { return math.Inf(1) - math.Inf(1) }
