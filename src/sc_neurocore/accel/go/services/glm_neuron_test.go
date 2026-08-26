// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service tests for GLMNeuron

package services

import (
	"math"
	"testing"
)

func TestGLMDeterministicForcedSpike(t *testing.T) {
	state := NewGLMNeuron(10, 20)
	spike, err := state.TryStep(20.0, 0.0)
	if err != nil || spike != 1 {
		t.Fatalf("forced spike expected: spike=%d err=%v", spike, err)
	}
	if state.SpikeBuf[0] != 1.0 || state.StimBuf[0] != 20.0 {
		t.Fatalf("history buffers not updated: %+v", state)
	}
}

func TestGLMReferenceFilters(t *testing.T) {
	state := NewGLMNeuron(10, 20)
	if math.Abs(state.K[0]-0.5) > 1e-15 {
		t.Fatalf("unexpected k[0]=%.17g", state.K[0])
	}
	if math.Abs(state.H[0]-(-4.5)) > 1e-15 {
		t.Fatalf("unexpected h[0]=%.17g", state.H[0])
	}
}

func TestGLMInvalidInputIsAtomic(t *testing.T) {
	state := NewGLMNeuron(10, 20)
	stimBefore := append([]float64(nil), state.StimBuf...)
	spikeBefore := append([]float64(nil), state.SpikeBuf...)
	if _, err := state.TryStep(math.NaN(), 0.5); err == nil {
		t.Fatal("NaN stimulus must fail")
	}
	if _, err := state.TryStep(1.0, 1.0); err == nil {
		t.Fatal("uniform >= 1 must fail")
	}
	if _, err := state.TryStep(1.0, math.NaN()); err == nil {
		t.Fatal("NaN uniform must fail")
	}
	for i := range stimBefore {
		if state.StimBuf[i] != stimBefore[i] {
			t.Fatal("invalid input mutated stimulus history")
		}
	}
	for i := range spikeBefore {
		if state.SpikeBuf[i] != spikeBefore[i] {
			t.Fatal("invalid input mutated spike history")
		}
	}
}

func TestGLMInvalidConfigurationIsAtomic(t *testing.T) {
	state := NewGLMNeuron(10, 20)
	state.Mu = math.NaN()
	if _, err := state.TryStep(1.0, 0.5); err == nil {
		t.Fatal("invalid configuration must fail")
	}
}

func TestGLMSimulateSeededRuns(t *testing.T) {
	_, spikesA := SimulateGLMNeuron(500, 5.0, 42)
	_, spikesB := SimulateGLMNeuron(500, 5.0, 42)
	if spikesA != spikesB {
		t.Fatalf("seeded simulation must be reproducible: %d vs %d", spikesA, spikesB)
	}
}

func TestGLMResetPreservesFilters(t *testing.T) {
	state := NewGLMNeuron(10, 20)
	if _, err := state.TryStep(5.0, 0.0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	k0 := state.K[0]
	state.Reset()
	if state.StimBuf[0] != 0.0 || state.SpikeBuf[0] != 0.0 || state.K[0] != k0 {
		t.Fatalf("unexpected reset state: %+v", state)
	}
}
