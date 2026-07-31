// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

package services

import (
	"math"
	"testing"
)

func TestPersistentNaNominalStep(t *testing.T) {
	state := NewPersistentNaNeuron()
	if spike, err := state.TryStep(5.0); err != nil || spike != 0 {
		t.Fatalf("unexpected step result: spike=%d err=%v", spike, err)
	}
	if state.V < -66.0 || state.V > -60.0 {
		t.Fatalf("unexpected voltage %.17g", state.V)
	}
}

func TestPersistentNaInvalidDriveIsAtomic(t *testing.T) {
	state := NewPersistentNaNeuron()
	before := *state
	if _, err := state.TryStep(math.NaN()); err == nil {
		t.Fatal("NaN drive must fail")
	}
	if *state != before {
		t.Fatal("invalid drive mutated state")
	}
}

func TestPersistentNaInvalidConfigurationIsAtomic(t *testing.T) {
	state := NewPersistentNaNeuron()
	state.CM = 0.0
	before := *state
	if _, err := state.TryStep(1.0); err == nil {
		t.Fatal("invalid configuration must fail")
	}
	if *state != before {
		t.Fatal("invalid configuration mutated state")
	}
}

func TestPersistentNaResetPreservesParameters(t *testing.T) {
	state := NewPersistentNaNeuron()
	state.GNap, state.V, state.P = 0.3, -80.0, 0.7
	state.Reset()
	if state.V != -65.0 || state.P != 0.0 || state.GNap != 0.3 {
		t.Fatalf("unexpected reset state: %+v", state)
	}
}
