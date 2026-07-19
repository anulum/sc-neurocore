// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for dual alpha-synapse LIF

package services

import (
	"math"
	"testing"
)

func TestAlphaCatalogueDefaults(t *testing.T) {
	state := NewAlphaNeuron()
	if !state.Valid() {
		t.Fatal("catalogue defaults must be valid")
	}
	if state.V != 0.0 || state.VThreshold != 1.0 || state.TauV != 20.0 ||
		state.TauExc != 5.0 || state.TauInh != 10.0 || state.Dt != 1.0 {
		t.Fatalf("unexpected catalogue defaults: %+v", state)
	}
}

func TestAlphaFilterMatchesExactCascade(t *testing.T) {
	riseNext, currentNext, err := alphaFilterCandidates(0.25, 0.1, 2.0, 5.0, 0.5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	steady := 5.0 * 2.0
	decay := math.Exp(-0.5 / 5.0)
	expectedRise := steady + (0.25-steady)*decay
	expectedCurrent := steady + decay*((0.1-steady)+(0.25-steady)*0.5/5.0)
	if math.Abs(riseNext-expectedRise) > 1.0e-12 || math.Abs(currentNext-expectedCurrent) > 1.0e-12 {
		t.Fatalf("exact cascade mismatch: got (%0.17g, %0.17g)", riseNext, currentNext)
	}
}

func TestAlphaDriveContributionHandlesEqualTimeConstants(t *testing.T) {
	exact, err := alphaDriveContribution(0.3, 0.2, 20.0, 20.0, 0.5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	rate := 1.0 / 20.0
	decay := math.Exp(-0.5 / 20.0)
	expected := rate * decay * (0.3*0.5 + 0.2*0.5*0.5/(2.0*20.0))
	if math.Abs(exact-expected) > 1.0e-12 {
		t.Fatalf("equal-tau convolution mismatch: got %0.17g want %0.17g", exact, expected)
	}
}

func TestAlphaSpikeResetsOnlyTheMembrane(t *testing.T) {
	state := &AlphaNeuronState{
		V: 0.9, AExc: 0.4, IExc: 0.6, AInh: 0.2, IInh: 0.1, VThreshold: 0.5,
		VRest: 0.0, TauV: 20.0, TauExc: 5.0, TauInh: 10.0, Dt: 1.0,
	}
	before := [4]float64{state.AExc, state.IExc, state.AInh, state.IInh}
	spike, err := state.Step(0.0, 0.0)
	if err != nil || spike != 1 {
		t.Fatalf("expected one spike, got spike=%d error=%v", spike, err)
	}
	if state.V != 0.0 {
		t.Fatalf("wrong membrane reset: %v", state.V)
	}
	decayExc := math.Exp(-1.0 / 5.0)
	decayInh := math.Exp(-1.0 / 10.0)
	if math.Abs(state.AExc-before[0]*decayExc) > 1.0e-12 ||
		math.Abs(state.IExc-decayExc*(before[1]+before[0]*1.0/5.0)) > 1.0e-12 ||
		math.Abs(state.AInh-before[2]*decayInh) > 1.0e-12 ||
		math.Abs(state.IInh-decayInh*(before[3]+before[2]*1.0/10.0)) > 1.0e-12 {
		t.Fatalf("cascade must not reset on spike: got (%v, %v, %v, %v)",
			state.AExc, state.IExc, state.AInh, state.IInh)
	}
}

func TestAlphaInvalidStatePreservesState(t *testing.T) {
	state := NewAlphaNeuron()
	state.V = 0.25
	state.AExc = 0.5
	before := [5]float64{state.V, state.AExc, state.IExc, state.AInh, state.IInh}
	state.Dt = 0.0
	if _, err := state.Step(0.5, 0.0); err == nil {
		t.Fatal("expected invalid-state error")
	}
	after := [5]float64{state.V, state.AExc, state.IExc, state.AInh, state.IInh}
	if after != before {
		t.Fatalf("state mutated on invalid input: got %v", after)
	}
}

func TestAlphaNonFiniteUpdatePreservesState(t *testing.T) {
	state := NewAlphaNeuron()
	state.V = -math.MaxFloat64
	before := [5]float64{state.V, state.AExc, state.IExc, state.AInh, state.IInh}
	if _, err := state.Step(math.MaxFloat64, 0.0); err == nil {
		t.Fatal("expected non-finite update error")
	}
	after := [5]float64{state.V, state.AExc, state.IExc, state.AInh, state.IInh}
	if after != before {
		t.Fatalf("state mutated on non-finite update: got %v", after)
	}
}

func TestAlphaLegacySimulationRejectsInvalidArguments(t *testing.T) {
	trace, spikes := SimulateAlphaNeuron(-1, 0.0, 0.0)
	if trace != nil || spikes != 0 {
		t.Fatalf("negative step count must fail closed: trace=%v spikes=%d", trace, spikes)
	}
	trace, spikes = SimulateAlphaNeuron(1, math.NaN(), 0.0)
	if trace != nil || spikes != 0 {
		t.Fatalf("non-finite current must fail closed: trace=%v spikes=%d", trace, spikes)
	}
}
