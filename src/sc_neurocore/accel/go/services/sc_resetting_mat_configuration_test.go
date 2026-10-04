// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — SC resetting-MAT configuration and reset contracts

package services

import (
	"math"
	"reflect"
	"testing"
)

// scResettingMATStateBits preserves exact dynamic and configuration bit patterns.
func scResettingMATStateBits(state *SCResettingMATNeuronState) [13]uint64 {
	var bits [13]uint64
	fields := reflect.ValueOf(state).Elem()
	for index := range bits {
		bits[index] = math.Float64bits(fields.Field(index).Float())
	}
	return bits
}

// TestSCResettingMATResetRefusalPreservesCompleteState checks refusal and configuration repair.
func TestSCResettingMATResetRefusalPreservesCompleteState(t *testing.T) {
	type invalid struct {
		field string
		value float64
	}
	cases := []invalid{
		{"VRest", -500.0}, {"VRest", 500.0},
		{"VReset", -201.0}, {"VReset", 101.0},
		{"TauM", 0.0}, {"TauM", -1.0}, {"Tau1", 0.0}, {"Tau1", -1.0},
		{"Tau2", 0.0}, {"Tau2", -1.0}, {"H1", -1.0}, {"H1", 1.0e9 + 1.0},
		{"H2", -1.0}, {"H2", 1.0e9 + 1.0}, {"Resistance", 0.0},
		{"Resistance", -1.0}, {"Dt", 0.0}, {"Dt", -1.0},
	}
	for _, field := range []string{"VRest", "VReset", "VThresholdBase", "TauM", "Tau1", "Tau2", "H1", "H2", "Resistance", "Dt"} {
		for _, value := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
			cases = append(cases, invalid{field, value})
		}
	}
	for _, bad := range cases {
		state := NewSCResettingMATNeuron()
		state.V, state.Theta1, state.Theta2 = -65.0, 2.0, 3.0
		reflect.ValueOf(state).Elem().FieldByName(bad.field).SetFloat(bad.value)
		before := scResettingMATStateBits(state)
		if err := state.TryReset(); err != ErrSCResettingMATInvalidReset {
			t.Fatalf("%s=%g accepted invalid reset: %v", bad.field, bad.value, err)
		}
		if after := scResettingMATStateBits(state); after != before {
			t.Fatalf("%s=%g reset refusal changed state", bad.field, bad.value)
		}
		state.Reset()
		if after := scResettingMATStateBits(state); after != before {
			t.Fatalf("%s=%g legacy reset changed refused state", bad.field, bad.value)
		}
		defaults := reflect.ValueOf(NewSCResettingMATNeuron()).Elem()
		reflect.ValueOf(state).Elem().FieldByName(bad.field).SetFloat(defaults.FieldByName(bad.field).Float())
		if err := state.TryReset(); err != nil {
			t.Fatalf("%s configuration recovery failed: %v", bad.field, err)
		}
		if state.V != -70.0 || state.Theta1 != 0.0 || state.Theta2 != 0.0 || state.Step(0.0) != 0 {
			t.Fatalf("%s did not recover the resting state: %+v", bad.field, *state)
		}
	}
}

// TestSCResettingMATResetRecoversInvalidDynamics requires recovery without configuration changes.
func TestSCResettingMATResetRecoversInvalidDynamics(t *testing.T) {
	for _, field := range []string{"V", "Theta1", "Theta2"} {
		for _, value := range []float64{math.NaN(), math.Inf(1), math.Inf(-1), -1.0e308, 1.0e308} {
			state := NewSCResettingMATNeuron()
			state.VRest, state.TauM, state.H1 = -65.0, 12.0, 4.0
			reflect.ValueOf(state).Elem().FieldByName(field).SetFloat(value)
			before := scResettingMATStateBits(state)
			if err := state.TryReset(); err != nil {
				t.Fatalf("%s=%g dynamic recovery failed: %v", field, value, err)
			}
			after := scResettingMATStateBits(state)
			for index := 3; index < len(before); index++ {
				if after[index] != before[index] {
					t.Fatalf("%s reset changed configuration field %d", field, index)
				}
			}
			if state.V != -65.0 || state.Theta1 != 0.0 || state.Theta2 != 0.0 || state.Step(0.0) != 0 {
				t.Fatalf("%s reset did not recover all dynamic fields: %+v", field, *state)
			}
		}
	}
}

// TestSCResettingMATResetEnvelopeBoundaries accepts every valid resting boundary.
func TestSCResettingMATResetEnvelopeBoundaries(t *testing.T) {
	for _, rest := range []float64{-200.0, -70.0, -65.0, 100.0} {
		state := NewSCResettingMATNeuron()
		state.VRest, state.Tau1, state.Tau2, state.Dt = rest, 20.0, 250.0, 0.25
		before := scResettingMATStateBits(state)
		state.Reset()
		if state.V != rest || state.Theta1 != 0.0 || state.Theta2 != 0.0 {
			t.Fatalf("valid resting boundary refused: %g", rest)
		}
		after := scResettingMATStateBits(state)
		for index := 3; index < len(before); index++ {
			if before[index] != after[index] {
				t.Fatalf("boundary reset changed configuration field %d", index)
			}
		}
	}
}
