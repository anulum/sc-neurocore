// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for the exact Wu et al. IQIF service

package services

import "testing"

func TestIQIFSourceTutorialTrace(t *testing.T) {
	trace, spikes, final, err := SimulateIQIFTrace(*NewIntegerQIFNeuron(), 400, 10)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	wantPrefix := []int64{138, 146, 153, 159, 165, 170, 176, 183, 190, 198, 207, 217, 229, 242, 128}
	for index, want := range wantPrefix {
		if trace[index] != want {
			t.Fatalf("trace[%d]=%d, want %d", index, trace[index], want)
		}
	}
	if spikes != 26 || final.V != 198 {
		t.Fatalf("got spikes=%d final=%d, want 26 and 198", spikes, final.V)
	}
}

func TestIQIFCompleteConfiguredContract(t *testing.T) {
	initial := IntegerQIFNeuronState{V: 100, VRest: 96, VThreshold: 180, VReset: 120, A: 3, B: 5, VMax: 240, VMin: 4}
	trace, spikes, final, err := SimulateIQIFTrace(initial, 128, 17)
	if err != nil || len(trace) != 128 {
		t.Fatalf("configured trace failed: len=%d err=%v", len(trace), err)
	}
	if spikes != 0 || final.V != 139 || final.V != trace[len(trace)-1] {
		t.Fatalf("configured result mismatch: spikes=%d final=%d, want 0 and 139", spikes, final.V)
	}
}

func TestIQIFStrictEventAndLowerClamp(t *testing.T) {
	state := NewIntegerQIFNeuron()
	state.V = state.VMax
	spike, err := state.Step(-6)
	if err != nil || spike != 0 || state.V != state.VMax {
		t.Fatalf("candidate equal to v_max must not spike: state=%+v spike=%d err=%v", state, spike, err)
	}
	spike, err = state.Step(-5)
	if err != nil || spike != 1 || state.V != state.VReset {
		t.Fatalf("candidate above v_max must spike and reset: state=%+v spike=%d err=%v", state, spike, err)
	}
	state.V = state.VMin
	spike, err = state.Step(-100)
	if err != nil || spike != 0 || state.V != state.VMin {
		t.Fatalf("lower clamp failed: state=%+v spike=%d err=%v", state, spike, err)
	}
}

func TestIQIFInvalidWorkDoesNotMutate(t *testing.T) {
	state := NewIntegerQIFNeuron()
	before := state.V
	state.A = -1
	if _, err := state.Step(10); err == nil {
		t.Fatal("expected invalid coefficient to fail")
	}
	if state.V != before {
		t.Fatalf("invalid work mutated state to %d", state.V)
	}
	if trace, spikes, _, err := SimulateIQIFTrace(*NewIntegerQIFNeuron(), -1, 10); err == nil || trace != nil || spikes != 0 {
		t.Fatalf("negative batch did not fail closed: trace=%v spikes=%d err=%v", trace, spikes, err)
	}
}

func TestIQIFResetPreservesParameters(t *testing.T) {
	state := IntegerQIFNeuronState{V: 140, VRest: 100, VThreshold: 180, VReset: 150, A: 2, B: 7, VMax: 250, VMin: 3}
	state.Reset()
	if state.V != 100 || state.VReset != 150 || state.A != 2 || state.B != 7 || state.VMax != 250 {
		t.Fatalf("reset corrupted the contract: %+v", state)
	}
}

func BenchmarkIQIFIntegerStep(b *testing.B) {
	state := NewIntegerQIFNeuron()
	spikes := 0
	for index := 0; index < b.N; index++ {
		spike, err := state.Step(10)
		if err != nil {
			b.Fatalf("unexpected error: %v", err)
		}
		spikes += spike
	}
	if !state.Valid() || spikes < 0 {
		b.Fatalf("invalid final state: %+v spikes=%d", state, spikes)
	}
}
