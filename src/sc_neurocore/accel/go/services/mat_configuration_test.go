// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source MAT complete reset and recovery contracts

package services

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

func matStateBits(s *MATNeuronState) [13]uint64 {
	values := []float64{s.V, s.Theta1, s.Theta2, s.RefractoryRemaining, s.Omega, s.TauM, s.Tau1, s.Tau2, s.Alpha1, s.Alpha2, s.Resistance, s.RefractoryPeriod, s.Dt}
	var result [13]uint64
	for index, value := range values {
		result[index] = math.Float64bits(value)
	}
	return result
}

// TestMATResetRefusalPreservesCompleteState uses the legacy public reset method.
func TestMATResetRefusalPreservesCompleteState(t *testing.T) {
	invalid := []struct {
		field string
		value float64
	}{
		{"Omega", -1e9 - 1}, {"Omega", 1e9 + 1},
		{"Alpha1", -1}, {"Alpha1", 1e9 + 1}, {"Alpha2", -1}, {"Alpha2", 1e9 + 1},
		{"TauM", 0}, {"TauM", -1}, {"Tau1", 0}, {"Tau1", -1},
		{"Tau2", 0}, {"Tau2", -1}, {"Resistance", 0}, {"Resistance", -1},
		{"RefractoryPeriod", -1}, {"Dt", 0}, {"Dt", -1},
	}
	for _, field := range []string{"Omega", "TauM", "Tau1", "Tau2", "Alpha1", "Alpha2", "Resistance", "RefractoryPeriod", "Dt"} {
		for _, value := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
			invalid = append(invalid, struct {
				field string
				value float64
			}{field, value})
		}
	}
	for _, sample := range invalid {
		state := NewMATNeuron()
		state.V, state.Theta1, state.Theta2, state.RefractoryRemaining = 20, 2, 3, 0.5
		reflect.ValueOf(state).Elem().FieldByName(sample.field).SetFloat(sample.value)
		before := matStateBits(state)
		if !errors.Is(state.TryReset(), ErrMATInvalidReset) || matStateBits(state) != before {
			t.Fatalf("%s=%v checked reset failed to preserve refused state", sample.field, sample.value)
		}
		state.Reset()
		if matStateBits(state) != before {
			t.Fatalf("%s=%v reset changed refused state", sample.field, sample.value)
		}
		reflect.ValueOf(state).Elem().FieldByName(sample.field).SetFloat(reflect.ValueOf(NewMATNeuron()).Elem().FieldByName(sample.field).Float())
		if err := state.TryReset(); err != nil {
			t.Fatalf("%s checked reset recovery failed: %v", sample.field, err)
		}
		state.Reset()
		if state.V != 0 || state.Theta1 != 0 || state.Theta2 != 0 || state.RefractoryRemaining != 0 || state.Step(0) != 0 {
			t.Fatalf("%s reset recovery failed", sample.field)
		}
	}
}

// TestMATResetRecoversCorruptedDynamics preserves all nine configuration fields.
func TestMATResetRecoversCorruptedDynamics(t *testing.T) {
	for _, field := range []string{"V", "Theta1", "Theta2", "RefractoryRemaining"} {
		for _, bad := range []float64{math.NaN(), math.Inf(1), math.Inf(-1), -1e308, 1e308} {
			state := NewMATNeuron()
			state.TauM, state.Alpha1, state.RefractoryPeriod = 8, 7, 0
			before := matStateBits(state)
			reflect.ValueOf(state).Elem().FieldByName(field).SetFloat(bad)
			if err := state.TryReset(); err != nil {
				t.Fatalf("%s checked dynamic recovery failed: %v", field, err)
			}
			state.Reset()
			after := matStateBits(state)
			for index := 4; index < len(before); index++ {
				if before[index] != after[index] {
					t.Fatalf("%s recovery changed configuration field %d", field, index)
				}
			}
			if state.V != 0 || state.Theta1 != 0 || state.Theta2 != 0 || state.RefractoryRemaining != 0 || state.Step(0) != 0 {
				t.Fatalf("%s dynamic recovery failed", field)
			}
		}
	}
}
