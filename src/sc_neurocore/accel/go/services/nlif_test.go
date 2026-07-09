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

func nlifReferenceRK4(state NonlinearLIFState, current float64) (float64, float64) {
	derivatives := func(v float64, w float64) (float64, float64) {
		nonlinear := state.A * (v - state.VRest) * (v - state.VCrit)
		return (nonlinear - w + current) / state.CM,
			(state.B*(v-state.VRest) - w) / state.TauW
	}
	k1v, k1w := derivatives(state.V, state.W)
	k2v, k2w := derivatives(state.V+0.5*state.DT*k1v, state.W+0.5*state.DT*k1w)
	k3v, k3w := derivatives(state.V+0.5*state.DT*k2v, state.W+0.5*state.DT*k2w)
	k4v, k4w := derivatives(state.V+state.DT*k3v, state.W+state.DT*k3w)
	nextV := state.V + (state.DT/6.0)*(k1v+2.0*k2v+2.0*k3v+k4v)
	nextW := state.W + (state.DT/6.0)*(k1w+2.0*k2w+2.0*k3w+k4w)
	return nextV, nextW
}

func TestNonlinearLIFStepMatchesRK4(t *testing.T) {
	state := DefaultNonlinearLIFState()
	wantV, wantW := nlifReferenceRK4(state, 20.0)
	if got := state.Step(20.0); got != 0 {
		t.Fatalf("first RK4 step should be subthreshold, got %d", got)
	}
	if math.Abs(state.V-wantV) > 1e-12 {
		t.Fatalf("V mismatch: got %.17g want %.17g", state.V, wantV)
	}
	if math.Abs(state.W-wantW) > 1e-12 {
		t.Fatalf("W mismatch: got %.17g want %.17g", state.W, wantW)
	}
}

func TestNonlinearLIFInvalidCurrentPreservesState(t *testing.T) {
	state := DefaultNonlinearLIFState()
	state.V = -60.0
	state.W = 0.5
	if got := state.Step(math.NaN()); got != -1 {
		t.Fatalf("invalid current must fail closed, got %d", got)
	}
	if state.V != -60.0 || state.W != 0.5 {
		t.Fatalf("invalid current mutated state: got (%v, %v)", state.V, state.W)
	}
}

func BenchmarkNonlinearLIFRK4(b *testing.B) {
	state := DefaultNonlinearLIFState()
	spikes := 0
	for i := 0; i < b.N; i++ {
		result := state.Step(20.0)
		if result < 0 {
			b.Fatalf("invalid RK4 step at iteration %d", i)
		}
		spikes += result
	}
	b.ReportMetric(float64(spikes), "spikes")
}
