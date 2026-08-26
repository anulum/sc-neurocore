// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service tests for NMDANeuron

package services

import (
	"math"
	"testing"
)

func TestNMDANominalStep(t *testing.T) {
	state := NewNMDANeuron()
	if spike, err := state.TryStep(5.0); err != nil || spike != 0 {
		t.Fatalf("unexpected step result: spike=%d err=%v", spike, err)
	}
	if math.Abs(state.V-(-63.155663780395777)) > 1e-12 {
		t.Fatalf("unexpected voltage %.17g", state.V)
	}
	if math.Abs(state.SNmda-0.025) > 1e-15 {
		t.Fatalf("unexpected s_nmda %.17g", state.SNmda)
	}
}

func TestNMDAInvalidDriveIsAtomic(t *testing.T) {
	state := NewNMDANeuron()
	before := *state
	if _, err := state.TryStep(math.NaN()); err == nil {
		t.Fatal("NaN drive must fail")
	}
	if _, err := state.TryStep(math.Inf(1)); err == nil {
		t.Fatal("+Inf drive must fail")
	}
	if _, err := state.TryStep(math.Inf(-1)); err == nil {
		t.Fatal("-Inf drive must fail")
	}
	if *state != before {
		t.Fatal("invalid drive mutated state")
	}
}

func TestNMDAInvalidConfigurationIsAtomic(t *testing.T) {
	state := NewNMDANeuron()
	state.CM = 0.0
	before := *state
	if _, err := state.TryStep(1.0); err == nil {
		t.Fatal("invalid configuration must fail")
	}
	if *state != before {
		t.Fatal("invalid configuration mutated state")
	}
}

func TestNMDASBuildsAndFiresWithInput(t *testing.T) {
	state := NewNMDANeuron()
	spikes := 0
	for i := 0; i < 2000; i++ {
		spike, err := state.TryStep(3.0)
		if err != nil {
			t.Fatalf("unexpected error at step %d: %v", i, err)
		}
		spikes += spike
	}
	if state.SNmda <= 0.0 {
		t.Fatalf("s_nmda must build with input, got %.17g", state.SNmda)
	}
	if spikes <= 5 {
		t.Fatalf("NMDA neuron must fire with input, got %d spikes", spikes)
	}
}

func TestNMDAResetPreservesParameters(t *testing.T) {
	state := NewNMDANeuron()
	state.GNmda, state.V, state.SNmda = 1.5, -30.0, 0.7
	state.Reset()
	if state.V != -65.0 || state.SNmda != 0.0 || state.GNmda != 1.5 {
		t.Fatalf("unexpected reset state: %+v", state)
	}
}
