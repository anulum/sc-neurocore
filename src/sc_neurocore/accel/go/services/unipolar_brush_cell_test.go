// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go unipolar brush cell service tests

package services

import (
	"math"
	"testing"
)

func unipolarBrushCellRelax(previous float64, steadyState float64, dt float64, tau float64) float64 {
	return previous + (steadyState-previous)*(-math.Expm1(-dt/tau))
}

func assertUnipolarBrushCellClose(t *testing.T, name string, got float64, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("%s = %.17g, want %.17g", name, got, want)
	}
}

func TestUnipolarBrushCellClosedFormPersistentAndMembraneRelaxation(t *testing.T) {
	state := NewUnipolarBrushCell()

	spike := state.Step(1.0)

	inputDrive := state.Gain * 1.0
	expectedPersistent := unipolarBrushCellRelax(0.0, state.PersistentGain*inputDrive, state.Dt, state.TauPersistent)
	expectedV := unipolarBrushCellRelax(
		state.VRest,
		state.VRest+inputDrive+expectedPersistent,
		state.Dt,
		state.TauM,
	)
	if spike != 0 {
		t.Fatalf("spike = %d, want 0", spike)
	}
	assertUnipolarBrushCellClose(t, "persistent", state.Persistent, expectedPersistent)
	assertUnipolarBrushCellClose(t, "v", state.V, expectedV)
}

func TestUnipolarBrushCellInvalidCurrentPreservesState(t *testing.T) {
	state := NewUnipolarBrushCell()
	state.V = -63.0
	state.Persistent = 2.0

	spike := state.Step(math.NaN())

	if spike != 0 {
		t.Fatalf("spike = %d, want 0", spike)
	}
	if state.V != -63.0 || state.Persistent != 2.0 {
		t.Fatalf("state mutated on invalid current: v=%v persistent=%v", state.V, state.Persistent)
	}
}

func TestUnipolarBrushCellCorruptedStatePreserved(t *testing.T) {
	state := NewUnipolarBrushCell()
	state.V = math.NaN()
	state.Persistent = 2.0

	spike := state.Step(10.0)

	if spike != 0 {
		t.Fatalf("spike = %d, want 0", spike)
	}
	if !math.IsNaN(state.V) || state.Persistent != 2.0 {
		t.Fatalf("corrupted state was not preserved: v=%v persistent=%v", state.V, state.Persistent)
	}
}
