// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go fidelity tests for the published DPI circuit

package services

import (
	"math"
	"reflect"
	"testing"
)

func configuredDPI() DPINeuronState {
	return DPINeuronState{
		IMem: 0.37, IAHP: 0.08, RefractoryTime: 0.0,
		IThreshold: 1.3, IReset: 0.2, IRest: 0.15,
		ITau: 0.9, IG: 1.4, ITauAHP: 0.12, IGA: 0.8,
		ISpike: 4.2, I0: 0.02, Kappa: 0.65, Alpha: 8.0,
		Tau: 7.0, TauAHP: 45.0, RefractoryPeriod: 0.6, Dt: 0.05,
	}
}

func TestDPIOneStepMatchesCoupledEulerReference(t *testing.T) {
	state := NewDPINeuron()
	spike, err := state.Step(5.0)
	if err != nil || spike != 0 {
		t.Fatalf("unexpected event result: spike=%d err=%v", spike, err)
	}
	if math.Abs(state.IMem-0.010201975272610835) > 1.0e-17 ||
		math.Abs(state.IAHP-0.00999) > 1.0e-17 || state.RefractoryTime != 0.0 {
		t.Fatalf("unexpected coupled state: %+v", state)
	}
}

func TestDPIThresholdAndRefractoryPulse(t *testing.T) {
	state := *NewDPINeuron()
	state.IMem = 0.99
	spike, err := state.Step(10.0)
	if err != nil || spike != 1 || state.IMem != state.IReset ||
		state.RefractoryTime != state.RefractoryPeriod {
		t.Fatalf("unexpected threshold crossing: spike=%d state=%+v err=%v", spike, state, err)
	}
	beforeAHP := state.IAHP
	spike, err = state.Step(0.0)
	if err != nil || spike != 0 || state.IMem != state.IReset || state.IAHP <= beforeAHP {
		t.Fatalf("unexpected refractory update: spike=%d state=%+v err=%v", spike, state, err)
	}
}

func TestDPIConfiguredAndEmptyContracts(t *testing.T) {
	initial := configuredDPI()
	trace, spikes, final, err := SimulateDPITrace(initial, 400, 5.0)
	if err != nil {
		t.Fatalf("unexpected configured error: %v", err)
	}
	if len(trace) != 400 || spikes != 4 || final.IMem != trace[len(trace)-1] ||
		final.IMem != 0.2 || math.Abs(final.IAHP-0.27412306389119817) > 2.0e-15 ||
		final.RefractoryTime != 0.0 {
		t.Fatalf("unexpected configured result: len=%d spikes=%d final=%+v", len(trace), spikes, final)
	}

	empty, emptySpikes, emptyFinal, err := SimulateDPITrace(initial, 0, 5.0)
	if err != nil || len(empty) != 0 || emptySpikes != 0 || !reflect.DeepEqual(emptyFinal, initial) {
		t.Fatalf("unexpected empty result: len=%d spikes=%d final=%+v err=%v", len(empty), emptySpikes, emptyFinal, err)
	}
}

func TestDPISimulateTraceMatchesFactoryEventVector(t *testing.T) {
	cases := []struct {
		current float64
		spikes  int
	}{
		{-0.1, 0}, {0.0, 0}, {1.0, 0}, {2.0, 0},
		{3.0, 1}, {5.0, 3}, {10.0, 6}, {20.0, 11}, {50.0, 21},
	}
	for _, test := range cases {
		trace, spikes, final, err := SimulateDPITrace(*NewDPINeuron(), 1_000, test.current)
		if err != nil {
			t.Fatalf("current=%v: unexpected error: %v", test.current, err)
		}
		if len(trace) != 1_000 || spikes != test.spikes || final.IMem != trace[len(trace)-1] {
			t.Fatalf("current=%v: len=%d spikes=%d final=%+v", test.current, len(trace), spikes, final)
		}
	}
}

func TestDPIInvalidAndOverflowContractsPreserveState(t *testing.T) {
	invalid := *NewDPINeuron()
	invalid.IGA = 0.0
	if invalid.Valid() {
		t.Fatal("zero adaptation saturation current accepted")
	}

	state := *NewDPINeuron()
	before := state
	if _, err := state.Step(math.NaN()); err != ErrDPIInvalidState {
		t.Fatalf("expected invalid-state error, got %v", err)
	}
	if !reflect.DeepEqual(state, before) {
		t.Fatalf("invalid current mutated state: before=%+v after=%+v", before, state)
	}

	state.Tau = math.SmallestNonzeroFloat64
	before = state
	if _, err := state.Step(math.MaxFloat64); err != ErrDPINonFiniteUpdate {
		t.Fatalf("expected non-finite update error, got %v", err)
	}
	if !reflect.DeepEqual(state, before) {
		t.Fatalf("overflow mutated state: before=%+v after=%+v", before, state)
	}
}

func TestDPIRejectedTraceContractsEmitNoState(t *testing.T) {
	valid := *NewDPINeuron()
	cases := []struct {
		state   DPINeuronState
		steps   int
		current float64
	}{
		{valid, -1, 0.0},
		{DPINeuronState{}, 1, 0.0},
		{valid, 1, math.Inf(1)},
		{valid, 1, -0.2},
	}
	for _, test := range cases {
		trace, spikes, final, err := SimulateDPITrace(test.state, test.steps, test.current)
		if err != ErrDPIInvalidState || trace != nil || spikes != 0 ||
			!reflect.DeepEqual(final, test.state) {
			t.Fatalf("rejected contract emitted state: trace=%v spikes=%d final=%+v err=%v", trace, spikes, final, err)
		}
	}
}

func TestDPIResetPreservesParameters(t *testing.T) {
	state := configuredDPI()
	parameters := state
	state.IMem = 0.75
	state.IAHP = 0.4
	state.RefractoryTime = 0.3
	state.Reset()
	parameters.IMem = parameters.IReset
	parameters.IAHP = parameters.I0
	parameters.RefractoryTime = 0.0
	if !reflect.DeepEqual(state, parameters) {
		t.Fatalf("reset changed parameters: got=%+v want=%+v", state, parameters)
	}
}

func TestDPISustainedDriveAdapts(t *testing.T) {
	state := NewDPINeuron()
	steps := make([]int, 0, 16)
	for index := 0; index < 5_000; index++ {
		spike, err := state.Step(5.0)
		if err != nil {
			t.Fatalf("unexpected error at %d: %v", index, err)
		}
		if spike == 1 {
			steps = append(steps, index)
		}
	}
	if len(steps) < 5 || steps[1]-steps[0] >= steps[len(steps)-1]-steps[len(steps)-2] {
		t.Fatalf("expected lengthening ISIs, got %v", steps)
	}
}

func BenchmarkDPICoupledEuler(b *testing.B) {
	state := NewDPINeuron()
	spikes := 0
	for index := 0; index < b.N; index++ {
		spike, err := state.Step(5.0)
		if err != nil {
			b.Fatalf("unexpected error: %v", err)
		}
		spikes += spike
	}
	if spikes < 0 {
		b.Fatal("unreachable negative spike count")
	}
}
