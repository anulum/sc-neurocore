// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for alpha service

package services

import (
	"math"
	"testing"
)

func alphaDriveContribution(
	currentDelta float64,
	riseDelta float64,
	tauDrive float64,
	tauV float64,
	dt float64,
) float64 {
	rateV := 1.0 / tauV
	rateDrive := 1.0 / tauDrive
	decayV := math.Exp(-dt / tauV)
	decayDrive := math.Exp(-dt / tauDrive)
	if math.Abs(rateV-rateDrive) <= 1.0e-14 {
		return rateV * decayV * (currentDelta*dt + riseDelta*dt*dt/(2.0*tauDrive))
	}
	rateDelta := rateV - rateDrive
	firstOrder := currentDelta * (decayDrive - decayV) / rateDelta
	secondOrder := riseDelta / tauDrive *
		(decayDrive*(rateDelta*dt-1.0) + decayV) / (rateDelta * rateDelta)
	return rateV * (firstOrder + secondOrder)
}

func alphaExactReference(
	state *AlphaNeuronState,
	excCurrent float64,
	inhCurrent float64,
) (float64, float64, float64, float64, float64) {
	aExcSS := state.TauExc * excCurrent
	aInhSS := state.TauInh * inhCurrent
	aExcDelta := state.AExc - aExcSS
	aInhDelta := state.AInh - aInhSS
	iExcDelta := state.IExc - aExcSS
	iInhDelta := state.IInh - aInhSS
	decayExc := math.Exp(-state.Dt / state.TauExc)
	decayInh := math.Exp(-state.Dt / state.TauInh)
	aExcNext := aExcSS + aExcDelta*decayExc
	aInhNext := aInhSS + aInhDelta*decayInh
	iExcNext := aExcSS + decayExc*(iExcDelta+aExcDelta*state.Dt/state.TauExc)
	iInhNext := aInhSS + decayInh*(iInhDelta+aInhDelta*state.Dt/state.TauInh)
	vSteady := state.VRest + aExcSS - aInhSS
	vNext := vSteady +
		(state.V-vSteady)*math.Exp(-state.Dt/state.TauV) +
		alphaDriveContribution(iExcDelta, aExcDelta, state.TauExc, state.TauV, state.Dt) -
		alphaDriveContribution(iInhDelta, aInhDelta, state.TauInh, state.TauV, state.Dt)
	return vNext, aExcNext, iExcNext, aInhNext, iInhNext
}

func TestAlphaExactLinearFlow(t *testing.T) {
	state := NewAlphaNeuron()
	state.V = 0.3
	state.AExc = 0.9
	state.IExc = 0.7
	state.AInh = 0.25
	state.IInh = 0.2
	state.VThreshold = 100.0
	state.Dt = 0.75
	expectedV, expectedAExc, expectedIExc, expectedAInh, expectedIInh := alphaExactReference(state, 0.8, 0.1)

	spike, err := state.Step(0.8, 0.1)
	if err != nil {
		t.Fatalf("unexpected step error: %v", err)
	}
	if spike != 0 {
		t.Fatalf("unexpected spike: %d", spike)
	}
	if math.Abs(state.V-expectedV) > 1.0e-12 {
		t.Fatalf("V mismatch: got %.17g want %.17g", state.V, expectedV)
	}
	if math.Abs(state.AExc-expectedAExc) > 1.0e-12 {
		t.Fatalf("AExc mismatch: got %.17g want %.17g", state.AExc, expectedAExc)
	}
	if math.Abs(state.IExc-expectedIExc) > 1.0e-12 {
		t.Fatalf("IExc mismatch: got %.17g want %.17g", state.IExc, expectedIExc)
	}
	if math.Abs(state.AInh-expectedAInh) > 1.0e-12 {
		t.Fatalf("AInh mismatch: got %.17g want %.17g", state.AInh, expectedAInh)
	}
	if math.Abs(state.IInh-expectedIInh) > 1.0e-12 {
		t.Fatalf("IInh mismatch: got %.17g want %.17g", state.IInh, expectedIInh)
	}
}

func TestAlphaInvalidStatePreservesState(t *testing.T) {
	state := NewAlphaNeuron()
	state.V = 0.25
	state.AExc = 0.6
	state.IExc = 0.5
	state.AInh = 0.2
	state.IInh = 0.125
	before := [5]float64{state.V, state.AExc, state.IExc, state.AInh, state.IInh}
	state.TauV = 0.0

	if _, err := state.Step(1.0, 0.5); err == nil {
		t.Fatal("expected invalid-state error")
	}
	after := [5]float64{state.V, state.AExc, state.IExc, state.AInh, state.IInh}
	if after != before {
		t.Fatalf("state mutated after invalid runtime state: got %v want %v", after, before)
	}
}
