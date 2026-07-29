// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for the Amari 1977 field mirror

package services

import (
	"math"
	"testing"
)

func TestAmariFieldVectorUpdateAndReset(t *testing.T) {
	state := NewAmariNeuralField()
	input := make([]float64, len(state.U))
	for index := range input {
		input[index] = 0.5
	}
	rate, err := state.Step(input)
	if err != nil || rate != 1.0 {
		t.Fatalf("unexpected first update: rate=%v err=%v", rate, err)
	}
	state.Reset()
	for _, value := range state.U {
		if value != 0.0 {
			t.Fatal("reset left nonzero state")
		}
	}
}

func TestAmariFieldFailureIsAtomic(t *testing.T) {
	state := NewAmariNeuralField()
	before := append([]float64{}, state.U...)
	input := make([]float64, len(state.U))
	input[3] = math.NaN()
	if _, err := state.Step(input); err == nil {
		t.Fatal("non-finite input was accepted")
	}
	for index := range before {
		if before[index] != state.U[index] {
			t.Fatal("failed update mutated state")
		}
	}
}

func BenchmarkAmariFieldStep(b *testing.B) {
	state := NewAmariNeuralField()
	input := make([]float64, len(state.U))
	for index := range input {
		input[index] = 0.2
	}
	b.ResetTimer()
	for range b.N {
		if _, err := state.Step(input); err != nil {
			b.Fatal(err)
		}
	}
}
