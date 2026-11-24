// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service tests for traub_miles

package services

import (
	"math"
	"testing"
)

func TestTraubMilesStepUsesRK4Reference(t *testing.T) {
	state := NewTraubMilesNeuron()
	state.V = -63.5
	state.M = 0.08
	state.H = 0.55
	state.N = 0.32

	spike, err := state.Step(4.0)
	if err != nil {
		t.Fatalf("unexpected Traub-Miles step error: %v", err)
	}

	if spike != 0 {
		t.Fatalf("unexpected spike from subthreshold RK4 reference point: %d", spike)
	}
	if math.Abs(state.V-(-65.6638958700765)) > 1.0e-13 {
		t.Fatalf("V RK4 reference mismatch: got %.17g", state.V)
	}
	if math.Abs(state.M-0.04237301812907925) > 1.0e-15 {
		t.Fatalf("M RK4 reference mismatch: got %.17g", state.M)
	}
	if math.Abs(state.H-0.5626824931070477) > 1.0e-15 {
		t.Fatalf("H RK4 reference mismatch: got %.17g", state.H)
	}
	if math.Abs(state.N-0.30356298261126924) > 1.0e-15 {
		t.Fatalf("N RK4 reference mismatch: got %.17g", state.N)
	}
	if math.Abs(state.V-(-65.66233161606698)) <= 1.0e-3 {
		t.Fatalf("state did not separate from Euler: V %.17g", state.V)
	}
}

func TestTraubMilesStepPreservesStateOnInvalidCurrent(t *testing.T) {
	state := NewTraubMilesNeuron()
	beforeV := state.V
	beforeM := state.M
	beforeH := state.H
	beforeN := state.N

	if _, err := state.Step(math.NaN()); err == nil {
		t.Fatal("expected invalid current error")
	}
	if state.V != beforeV || state.M != beforeM || state.H != beforeH || state.N != beforeN {
		t.Fatalf("state mutated after invalid current: %.17g %.17g %.17g %.17g", state.V, state.M, state.H, state.N)
	}
}

// traubMilesRate returns rate number index of traubMilesRates(v), failing the test on an error.
func traubMilesRate(t *testing.T, v float64, index int) float64 {
	t.Helper()
	am, bm, ah, bh, an, bn, err := traubMilesRates(v)
	if err != nil {
		t.Fatalf("rates(%v) failed: %v", v, err)
	}
	return []float64{am, bm, ah, bh, an, bn}[index]
}

// TestTraubMilesRatesTakeTheAnalyticLimitsAtTheRemovableSingularities checks that alpha_m, beta_m
// and alpha_n, which are 0/0 at -54, -27 and -52 mV, take their limits a*k inside the 1e-6 window
// and stay continuous across its edge.
func TestTraubMilesRatesTakeTheAnalyticLimitsAtTheRemovableSingularities(t *testing.T) {
	cases := []struct {
		index int
		v0    float64
		limit float64
	}{{0, -54.0, 1.28}, {1, -27.0, 1.40}, {4, -52.0, 0.16}}
	for _, c := range cases {
		for _, offset := range []float64{0.0, 1e-12, -1e-9, 1e-7, -5e-7, 1e-6} {
			rate := traubMilesRate(t, c.v0+offset, c.index)
			if math.Abs(rate-c.limit) > 0.5*math.Abs(offset)+1e-15 {
				t.Fatalf("rate %d at %v = %.17g, limit %.17g", c.index, c.v0+offset, rate, c.limit)
			}
		}
		// Across the window edge the rate moves by no more than the closed form's rounding.
		for _, sign := range []float64{-1.0, 1.0} {
			inside := traubMilesRate(t, c.v0+sign*0.99e-6, c.index)
			outside := traubMilesRate(t, c.v0+sign*1.01e-6, c.index)
			if math.Abs(outside-inside) > 0.5*0.02e-6+1e-9 {
				t.Fatalf("rate %d jumps by %.3g across the window edge", c.index, outside-inside)
			}
		}
	}
}
