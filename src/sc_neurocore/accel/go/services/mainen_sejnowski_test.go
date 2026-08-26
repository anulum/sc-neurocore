// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service tests for MainenSejnowskiNeuron

package services

import (
	"math"
	"testing"
)

func TestMainenNominalStep(t *testing.T) {
	state := NewMainenSejnowskiNeuron()
	if spike, err := state.TryStep(10.0); err != nil || spike != 0 {
		t.Fatalf("unexpected step result: spike=%d err=%v", spike, err)
	}
	if math.Abs(state.Vs-(-32.668480035293555)) > 1e-12 {
		t.Fatalf("unexpected vs %.17g", state.Vs)
	}
	if math.Abs(state.H-0.6581322365920295) > 1e-12 {
		t.Fatalf("unexpected h %.17g", state.H)
	}
}

func TestMainenRateLimitsAtSingularVoltages(t *testing.T) {
	if mainenLinoid(0.0, 9.0) != 9.0 {
		t.Fatal("linoid limit at zero must be k")
	}
	if math.Abs(mainenLinoid(1e-9, 5.0)-5.0) > 1e-8 {
		t.Fatal("linoid must be continuous near zero")
	}
	for _, vSingular := range []float64{-25.0, -40.0, -65.0, 20.0} {
		exact := NewMainenSejnowskiNeuron()
		exact.Va = vSingular
		near := NewMainenSejnowskiNeuron()
		near.Va = vSingular + 1e-9
		if _, err := exact.TryStep(0.0); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if _, err := near.TryStep(0.0); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		delta := math.Max(math.Abs(exact.Vs-near.Vs), math.Abs(exact.Va-near.Va))
		if delta > 1e-6 {
			t.Fatalf("step must be continuous at va=%v, delta=%g", vSingular, delta)
		}
	}
}

func TestMainenInvalidDriveIsAtomic(t *testing.T) {
	state := NewMainenSejnowskiNeuron()
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

func TestMainenInvalidConfigurationIsAtomic(t *testing.T) {
	state := NewMainenSejnowskiNeuron()
	state.CS = 0.0
	before := *state
	if _, err := state.TryStep(1.0); err == nil {
		t.Fatal("invalid configuration must fail")
	}
	if *state != before {
		t.Fatal("invalid configuration mutated state")
	}
}

func TestMainenResetPreservesParameters(t *testing.T) {
	state := NewMainenSejnowskiNeuron()
	state.Kappa, state.Vs = 20.0, -30.0
	state.Reset()
	if state.Vs != -65.0 || state.Kappa != 20.0 {
		t.Fatalf("unexpected reset state: %+v", state)
	}
}
