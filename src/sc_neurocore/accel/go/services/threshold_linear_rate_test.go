// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for configurable threshold-linear rate

package services

import (
	"math"
	"testing"
)

func TestThresholdLinearRateConfiguredBranches(t *testing.T) {
	state := ThresholdLinearRateNeuronState{R: 0.25, Theta: 1.5, Gain: 2.0}
	for _, row := range []struct {
		current float64
		want    float64
	}{{1.0, 0.0}, {1.5, 0.0}, {3.0, 3.0}} {
		got, err := state.Step(row.current)
		if err != nil || got != row.want {
			t.Fatalf("current %.17g: got %.17g, %v; want %.17g", row.current, got, err, row.want)
		}
	}
}

func TestThresholdLinearRateConfiguredBatchMatchesGolden(t *testing.T) {
	initial := ThresholdLinearRateNeuronState{R: 0.25, Theta: 1.5, Gain: 2.0}
	trace, final, err := SimulateThresholdLinearRateTrace(initial, 6, 3.0)
	if err != nil {
		t.Fatalf("unexpected batch error: %v", err)
	}
	for index, value := range trace {
		if value != 3.0 {
			t.Fatalf("trace[%d] = %.17g, want 3", index, value)
		}
	}
	if final.R != 3.0 || final.Theta != 1.5 || final.Gain != 2.0 {
		t.Fatalf("unexpected final state: %#v", final)
	}
}

func TestThresholdLinearRateBatchRejectsInvalidContractAtomically(t *testing.T) {
	initial := ThresholdLinearRateNeuronState{R: 0.25, Theta: 1.5, Gain: 2.0}
	trace, final, err := SimulateThresholdLinearRateTrace(initial, -1, 3.0)
	if err == nil || trace != nil || final != initial {
		t.Fatal("negative step count did not fail atomically")
	}
	trace, final, err = SimulateThresholdLinearRateTrace(initial, 1, math.NaN())
	if err == nil || trace != nil || final != initial {
		t.Fatal("non-finite input did not fail atomically")
	}
	overflow := ThresholdLinearRateNeuronState{R: 0.25, Theta: 0.0, Gain: 1.0e308}
	trace, final, err = SimulateThresholdLinearRateTrace(overflow, 1, 1.0e308)
	if err == nil || trace != nil || final != overflow {
		t.Fatal("overflowing transfer did not fail atomically")
	}
}

func TestThresholdLinearRateResetPreservesConfiguration(t *testing.T) {
	state := ThresholdLinearRateNeuronState{R: 0.25, Theta: -0.4, Gain: 2.5}
	state.Reset()
	if state != (ThresholdLinearRateNeuronState{R: 0.0, Theta: -0.4, Gain: 2.5}) {
		t.Fatalf("reset changed configuration: %#v", state)
	}
}

func TestThresholdLinearRateHistoricalHelperDoesNotCountRatesAsSpikes(t *testing.T) {
	trace, spikes := SimulateThresholdLinearRateNeuron(4, 3.0)
	if len(trace) != 4 || spikes != 0 {
		t.Fatalf("unexpected historical helper result: len=%d spikes=%d", len(trace), spikes)
	}
}
