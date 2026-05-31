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

func exactRelaxLugaro(value float64, target float64, tau float64, dt float64) float64 {
	return target + (value-target)*math.Exp(-dt/tau)
}

func assertLugaroClose(t *testing.T, name string, got float64, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("%s mismatch: got %.16e want %.16e", name, got, want)
	}
}

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

func TestLugaroCellInvalidVoltagePreservesState(t *testing.T) {
	cell := NewLugaroCell()
	cell.V = 60.1
	beforeV := cell.V
	beforeAdapt := cell.Adapt

	if spike := cell.Step(5.0); spike != 0 {
		t.Fatalf("invalid voltage must not spike, got %d", spike)
	}
	if cell.V != beforeV || cell.Adapt != beforeAdapt {
		t.Fatalf("invalid voltage mutated state")
	}
}

func TestLugaroCellClosedFormMembraneAndAdaptationRelaxation(t *testing.T) {
	cell := NewLugaroCell()
	cell.V = -56.0
	cell.Adapt = 0.2
	cell.Gain = 0.0

	vInf := cell.VRest - cell.Adapt
	expectedV := exactRelaxLugaro(cell.V, vInf, cell.TauM, cell.Dt)
	adaptInf := math.Max(0.0, cell.AAdapt*math.Max(0.0, expectedV-cell.VRest))
	expectedAdapt := math.Max(0.0, exactRelaxLugaro(cell.Adapt, adaptInf, cell.TauAdapt, cell.Dt))

	if spike := cell.Step(0.0); spike != 0 {
		t.Fatalf("subthreshold exact relaxation must not spike, got %d", spike)
	}
	assertLugaroClose(t, "v", cell.V, expectedV)
	assertLugaroClose(t, "adapt", cell.Adapt, expectedAdapt)
}
