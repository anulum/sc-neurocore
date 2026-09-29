// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go public dense parameter admission

package conversion_test

import (
	c "github.com/anulum/sc-neurocore/accel/conversion"
	"math"
	"testing"
)

func TestPublicParameterAdmission(t *testing.T) {
	for _, entry := range []struct {
		inputs, outputs int
		weights, bias   []float64
		theta, fraction float64
		budget          int
		want            error
	}{
		{0, 1, nil, nil, 1, 0, 1024, c.ErrInvalidInput},
		{1, 1, []float64{1}, []float64{0, 0}, 1, 0, 1024, c.ErrInvalidInput},
		{1, 1, []float64{math.NaN()}, nil, 1, 0, 1024, c.ErrInvalidInput},
		{1, 1, []float64{1}, []float64{math.Inf(1)}, 1, 0, 1024, c.ErrInvalidInput},
		{1, 1, []float64{1}, nil, 0, 0, 1024, c.ErrInvalidInput},
		{1, 1, []float64{1}, nil, 1, math.NaN(), 1024, c.ErrInvalidInput},
		{1, 1, []float64{1}, nil, 1, 0, 0, c.ErrInvalidInput},
		{1, 1, []float64{1}, nil, 1, 0, 15, c.ErrResourceLimit},
	} {
		if _, err := c.NewDenseLayer(entry.inputs, entry.outputs, entry.weights, entry.bias, entry.theta, entry.fraction, entry.budget); err != entry.want {
			t.Fatal(entry, err)
		}
	}
	first, _ := c.NewDenseLayer(1, 1, []float64{1}, nil, 1, 0, 1024)
	second, _ := c.NewDenseLayer(2, 1, []float64{1, 1}, nil, 1, 0, 1024)
	for _, entry := range []struct {
		layers []c.DenseLayer
		mode   c.OutputMode
		budget int
		want   error
	}{
		{nil, c.Spikes, 1024, c.ErrInvalidInput},
		{[]c.DenseLayer{first}, "invalid", 1024, c.ErrInvalidInput},
		{[]c.DenseLayer{{}}, c.Spikes, 1024, c.ErrInvalidInput},
		{[]c.DenseLayer{first, second}, c.Spikes, 1024, c.ErrInvalidInput},
		{[]c.DenseLayer{first}, c.Spikes, 15, c.ErrResourceLimit},
	} {
		if _, err := c.NewConvertedSNN(entry.layers, entry.mode, entry.budget); err != entry.want {
			t.Fatal(entry, err)
		}
	}
}
