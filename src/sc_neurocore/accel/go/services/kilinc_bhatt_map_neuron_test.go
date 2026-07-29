// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Kilinc-Bhatt Go recurrence tests

package services

import (
	"math"
	"testing"
)

func TestKilincBhattNominalDynamics(t *testing.T) {
	state := NewKilincBhattMapNeuron()
	spikes := 0
	for step := 0; step < 500; step++ {
		spike, err := state.TryStep(1.0)
		if err != nil {
			t.Fatalf("valid step %d failed: %v", step, err)
		}
		spikes += spike
	}
	if spikes == 0 || !state.Valid() {
		t.Fatalf("expected bounded spiking dynamics, spikes=%d state=%+v", spikes, state)
	}
}

func TestKilincBhattRejectsNonFiniteDriveAtomically(t *testing.T) {
	state := NewKilincBhattMapNeuron()
	beforeX, beforeTheta := state.X, state.Theta
	if _, err := state.TryStep(math.NaN()); err == nil {
		t.Fatal("NaN drive must fail")
	}
	if state.X != beforeX || state.Theta != beforeTheta {
		t.Fatalf("invalid drive mutated state: %+v", state)
	}
}

func TestKilincBhattRejectsInvalidConfigurationAtomically(t *testing.T) {
	state := NewKilincBhattMapNeuron()
	state.Beta = 1.1
	beforeX, beforeTheta := state.X, state.Theta
	if _, err := state.TryStep(1.0); err == nil {
		t.Fatal("invalid beta must fail")
	}
	if state.X != beforeX || state.Theta != beforeTheta {
		t.Fatalf("invalid configuration mutated state: %+v", state)
	}
}

func TestKilincBhattStepFailsClosed(t *testing.T) {
	state := NewKilincBhattMapNeuron()
	if spike := state.Step(math.Inf(1)); spike != 0 {
		t.Fatalf("invalid drive produced spike %d", spike)
	}
	if state.X != 0.0 || state.Theta != 0.0 {
		t.Fatalf("invalid drive mutated state: %+v", state)
	}
}
