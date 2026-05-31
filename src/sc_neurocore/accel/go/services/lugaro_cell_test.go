// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go LugaroCell behavioural tests

package services

import (
	"math"
	"testing"
)

func TestLugaroCellStepPreservesBoundsAndAdaptation(t *testing.T) {
	cell := NewLugaroCell()

	for i := 0; i < 200; i++ {
		spike := cell.Step(0.0)
		if spike != 0 && spike != 1 {
			t.Fatalf("spike indicator must be binary, got %d", spike)
		}
	}

	if math.IsNaN(cell.V) || math.IsInf(cell.V, 0) || cell.V < -100.0 || cell.V > 60.0 {
		t.Fatalf("membrane potential must stay finite and bounded, got %f", cell.V)
	}
	if math.IsNaN(cell.Adapt) || math.IsInf(cell.Adapt, 0) || cell.Adapt < 0.0 {
		t.Fatalf("adaptation must stay finite and non-negative, got %f", cell.Adapt)
	}
}

func TestLugaroCellSerotoninRaisesFiring(t *testing.T) {
	without := NewLugaroCell()
	with := NewLugaroCell()
	with.Serotonin = 1.0
	spikesWithout := 0
	spikesWith := 0

	for i := 0; i < 2000; i++ {
		spikesWithout += without.Step(3.0)
		spikesWith += with.Step(3.0)
	}

	if spikesWith < spikesWithout {
		t.Fatalf("serotonin should not reduce firing: with=%d without=%d", spikesWith, spikesWithout)
	}
}

func TestLugaroCellInvalidDrivePreservesState(t *testing.T) {
	cell := NewLugaroCell()
	beforeV := cell.V
	beforeAdapt := cell.Adapt

	if spike := cell.Step(math.NaN()); spike != 0 {
		t.Fatalf("invalid drive must not spike, got %d", spike)
	}
	if cell.V != beforeV || cell.Adapt != beforeAdapt {
		t.Fatalf("invalid drive mutated state")
	}
}

func TestLugaroCellCorruptedStatePreservesState(t *testing.T) {
	cell := NewLugaroCell()
	cell.Adapt = math.NaN()
	beforeV := cell.V
	beforeAdapt := cell.Adapt

	if spike := cell.Step(5.0); spike != 0 {
		t.Fatalf("corrupted state must not spike, got %d", spike)
	}
	if cell.V != beforeV || !math.IsNaN(beforeAdapt) || !math.IsNaN(cell.Adapt) {
		t.Fatalf("corrupted state mutated during fail-closed step")
	}
}
