// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for poisson

package services

import (
	"math"
	"testing"
)

func TestPoissonStepSaturatesOnlyAtBoundedProbabilityOne(t *testing.T) {
	state := NewPoissonNeuron()

	spike, err := state.Step(1.0e9)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spike != 1 {
		t.Fatalf("expected saturated high-rate spike, got %d", spike)
	}
}

func TestPoissonRejectsInvalidRuntimeState(t *testing.T) {
	state := NewPoissonNeuron()
	state.DtMs = 0.0

	if _, err := state.Step(-1.0); err == nil {
		t.Fatalf("expected invalid timestep to fail")
	}
}

func TestPoissonRejectsNonFiniteRateOverride(t *testing.T) {
	state := NewPoissonNeuron()

	if _, err := state.Step(math.Inf(1)); err == nil {
		t.Fatalf("expected non-finite rate override to fail")
	}
}

func TestPoissonRejectsNonFiniteIntervalHazard(t *testing.T) {
	state := NewPoissonNeuron()
	state.RateHz = 1.0e308
	state.DtMs = 1.0e308

	if _, err := state.Step(-1.0); err == nil {
		t.Fatalf("expected overflowing interval hazard to fail")
	}
}
