// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go upper motor neuron service tests

package services

import (
	"math"
	"testing"
)

func upperMotorNeuronClose(t *testing.T, name string, got float64, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("%s = %.17g, want %.17g", name, got, want)
	}
}

func upperMotorTestGate(previous float64, alpha float64, beta float64, dt float64) float64 {
	total := alpha + beta
	steady := alpha / total
	return math.Min(1.0, math.Max(0.0, steady+(previous-steady)*math.Exp(-total*dt)))
}

func upperMotorTestGateInf(previous float64, steady float64, tau float64, dt float64) float64 {
	return math.Min(1.0, math.Max(0.0, steady+(previous-steady)*math.Exp(-dt/tau)))
}

func upperMotorReferenceStep(state UpperMotorNeuronState, current float64) UpperMotorNeuronState {
	vt := -56.2
	for step := 0; step < 4; step++ {
		dv := state.V - vt
		xM := dv - 13.0
		alphaM := 0.32 * 4.0
		if math.Abs(xM) >= 1e-6 {
			alphaM = -0.32 * xM / (math.Exp(-xM/4.0) - 1.0)
		}
		xH := dv - 17.0
		betaM := 0.28 * 5.0
		if math.Abs(xH) >= 1e-6 {
			betaM = 0.28 * xH / (math.Exp(xH/5.0) - 1.0)
		}
		alphaH := 0.128 * math.Exp(-(dv-17.0)/18.0)
		betaH := 4.0 / (1.0 + math.Exp(-(dv-40.0)/5.0))
		xN := dv - 15.0
		alphaN := 0.032 * 5.0
		if math.Abs(xN) >= 1e-6 {
			alphaN = -0.032 * xN / (math.Exp(-xN/5.0) - 1.0)
		}
		betaN := 0.5 * math.Exp(-(dv-10.0)/40.0)
		state.M = upperMotorTestGate(state.M, alphaM, betaM, state.Dt)
		state.H = upperMotorTestGate(state.H, alphaH, betaH, state.Dt)
		state.N = upperMotorTestGate(state.N, alphaN, betaN, state.Dt)
		pInf := 1.0 / (1.0 + math.Exp(-(state.V+35.0)/10.0))
		tauP := 400.0 / (3.3*math.Exp((state.V+35.0)/20.0) + math.Exp(-(state.V+35.0)/20.0))
		state.P = upperMotorTestGateInf(state.P, pInf, tauP, state.Dt)
		sInf := 1.0 / (1.0 + math.Exp(-(state.V+20.0)/5.0))
		state.S = upperMotorTestGateInf(state.S, sInf, 10.0, state.Dt)
		gNa := state.GNa * state.M * state.M * state.M * state.H
		gK := state.GK * math.Pow(state.N, 4)
		gM := state.GM * state.P
		gCa := state.GCa * state.S * state.S
		gTotal := gNa + gK + gM + gCa + state.GL
		steadyV := (current + gNa*state.ENa + gK*state.EK + gM*state.EK + gCa*state.ECa + state.GL*state.EL) / gTotal
		state.V = steadyV + (state.V-steadyV)*math.Exp(-(gTotal/state.CM)*state.Dt)
	}
	return state
}

func TestUpperMotorNeuronConductanceStepAndGates(t *testing.T) {
	state := NewUpperMotorNeuron()
	expected := upperMotorReferenceStep(*state, 5.0)

	spike := state.Step(5.0)

	if spike != 0 {
		t.Fatalf("spike = %d, want 0", spike)
	}
	upperMotorNeuronClose(t, "v", state.V, expected.V)
	upperMotorNeuronClose(t, "m", state.M, expected.M)
	upperMotorNeuronClose(t, "h", state.H, expected.H)
	upperMotorNeuronClose(t, "n", state.N, expected.N)
	upperMotorNeuronClose(t, "p", state.P, expected.P)
	upperMotorNeuronClose(t, "s", state.S, expected.S)
}

func TestUpperMotorNeuronInvalidCurrentPreservesState(t *testing.T) {
	state := NewUpperMotorNeuron()
	state.V = -64.0
	state.M = 0.1
	state.H = 0.7
	state.N = 0.2
	state.P = 0.1
	state.S = 0.2

	spike := state.Step(math.NaN())

	if spike != 0 {
		t.Fatalf("spike = %d, want 0", spike)
	}
	if state.V != -64.0 || state.M != 0.1 || state.H != 0.7 || state.N != 0.2 || state.P != 0.1 || state.S != 0.2 {
		t.Fatalf("state mutated on invalid current: %+v", state)
	}
}

func TestUpperMotorNeuronCorruptedGatePreserved(t *testing.T) {
	state := NewUpperMotorNeuron()
	state.M = 1.5
	before := *state

	spike := state.Step(5.0)

	if spike != 0 {
		t.Fatalf("spike = %d, want 0", spike)
	}
	if *state != before {
		t.Fatalf("corrupted state mutated: got %+v want %+v", state, before)
	}
}
