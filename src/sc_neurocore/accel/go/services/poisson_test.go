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

func TestPoissonSeededSequenceAndResetAreReplayable(t *testing.T) {
	state := NewPoissonNeuronWithSeed(42)
	first := make([]int, 4096)
	for index := range first {
		spike, err := state.Step(250.0)
		if err != nil {
			t.Fatalf("step %d failed: %v", index, err)
		}
		first[index] = spike
	}
	finalRNG := state.RNGState
	state.Reset()
	for index, expected := range first {
		spike, err := state.Step(250.0)
		if err != nil {
			t.Fatalf("replay step %d failed: %v", index, err)
		}
		if spike != expected {
			t.Fatalf("event mismatch at %d: got %d want %d", index, spike, expected)
		}
	}
	if state.RNGState != finalRNG {
		t.Fatalf("final RNG mismatch: got %04x want %04x", state.RNGState, finalRNG)
	}
}

func TestPoissonFullPeriodExactQuarterHazardCount(t *testing.T) {
	initial := *NewPoissonNeuronWithSeed(0xACE1)
	events, final, err := SimulatePoissonTrace(initial, 65535, 250.0)
	if err != nil {
		t.Fatalf("full-period simulation failed: %v", err)
	}
	spikes := 0
	for _, event := range events {
		spikes += int(event)
	}
	if spikes != 14496 {
		t.Fatalf("spike count = %d, want 14496", spikes)
	}
	if final.RNGState != 0xACE1 {
		t.Fatalf("final RNG = %04x, want ace1", final.RNGState)
	}
	if initial.RNGState != 0xACE1 {
		t.Fatal("batch mutated the caller's value")
	}
}

func TestPoissonZeroRateStillConsumesOneCanonicalTrial(t *testing.T) {
	state := NewPoissonNeuronWithSeed(0xACE1)
	before := state.RNGState
	spike, err := state.Step(0.0)
	if err != nil {
		t.Fatalf("zero-rate step failed: %v", err)
	}
	if spike != 0 {
		t.Fatalf("zero rate emitted spike %d", spike)
	}
	if state.RNGState == before {
		t.Fatal("zero-probability trial did not advance the RNG")
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
