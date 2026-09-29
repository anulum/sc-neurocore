// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go borrowed coefficient constructor acceptance

package conversion_test

import (
	"github.com/anulum/sc-neurocore/accel/conversion"
	"math"
	"testing"
)

func TestBorrowedCoefficientConstructionOwnsCompleteStack(t *testing.T) {
	weights := []float64{1}
	bias := []float64{0.25}
	source := []conversion.LayerParameters{{Inputs: 1, Outputs: 1, Weights: weights, Bias: bias, Threshold: 1, InitialFraction: 0.5}}
	model, err := conversion.NewConvertedSNNFromParameters(source, conversion.Spikes, 32)
	if err != nil {
		t.Fatal(err)
	}
	weights[0] = 99
	bias[0] = 99
	source[0].Threshold = 99
	result, err := model.Replay([]float64{1}, 1, 1, conversion.ReplayOptions{Trace: true, BinaryInputs: true, MaxWorkingBytes: 1024})
	if err != nil || result.Output[0] != 1 || result.FinalState[0][0] != 0.75 {
		t.Fatalf("owned model: %+v %v", result, err)
	}
	initial := []conversion.LayerParameters{{Inputs: 1, Outputs: 1, Weights: []float64{1}, Threshold: 1}}
	linear, err := conversion.NewConvertedSNNFromParameters(initial, conversion.Linear, 16)
	if err != nil {
		t.Fatal(err)
	}
	got, err := linear.Replay([]float64{1}, 1, 1, conversion.ReplayOptions{MaxWorkingBytes: 1024})
	if err != nil || got.Output[0] != 1 {
		t.Fatal(got, err)
	}
}

func TestBorrowedCoefficientConstructorRefusesWithoutModel(t *testing.T) {
	valid := conversion.LayerParameters{Inputs: 1, Outputs: 1, Weights: []float64{1}, Threshold: 1}
	cases := []struct {
		source []conversion.LayerParameters
		mode   conversion.OutputMode
		budget int
		want   error
	}{
		{nil, conversion.Spikes, 16, conversion.ErrInvalidInput},
		{[]conversion.LayerParameters{valid}, "unknown", 16, conversion.ErrInvalidInput},
		{[]conversion.LayerParameters{valid}, conversion.Spikes, 0, conversion.ErrInvalidInput},
		{[]conversion.LayerParameters{{Inputs: 0, Outputs: 1}}, conversion.Spikes, 16, conversion.ErrInvalidInput},
		{[]conversion.LayerParameters{{Inputs: 1, Outputs: 0}}, conversion.Spikes, 16, conversion.ErrInvalidInput},
		{[]conversion.LayerParameters{valid, {Inputs: 2, Outputs: 1, Weights: []float64{1, 1}, Threshold: 1}}, conversion.Spikes, 64, conversion.ErrInvalidInput},
		{[]conversion.LayerParameters{valid}, conversion.Spikes, 15, conversion.ErrResourceLimit},
		{[]conversion.LayerParameters{{Inputs: 1, Outputs: 1, Weights: []float64{math.NaN()}, Threshold: 1}}, conversion.Spikes, 16, conversion.ErrInvalidInput},
		{[]conversion.LayerParameters{{Inputs: 1, Outputs: 1, Weights: []float64{1}, Bias: []float64{math.Inf(1)}, Threshold: 1}}, conversion.Spikes, 32, conversion.ErrInvalidInput},
	}
	for _, c := range cases {
		model, err := conversion.NewConvertedSNNFromParameters(c.source, c.mode, c.budget)
		if model != nil || err != c.want {
			t.Fatalf("unexpected model/error: %v %v; want %v", model, err, c.want)
		}
	}
}
