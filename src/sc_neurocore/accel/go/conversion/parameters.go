// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go owned dense IF parameters

// Package conversion implements dense-if-f64-sequential-v1 explicit-frame replay.
package conversion

import (
	"errors"
	"math"
	"math/big"
)

// ErrInvalidInput reports invalid shape, domain or finite coefficients.
var ErrInvalidInput = errors.New("invalid dense IF input or parameters")

// ErrOverflow reports finite arithmetic producing nonfinite current or state.
var ErrOverflow = errors.New("dense IF arithmetic overflow")

// ErrResourceLimit reports numeric buffers exceeding the working byte budget.
var ErrResourceLimit = errors.New("dense IF numeric resource limit")

// OutputMode declares the final response semantics.
type OutputMode string

// Spikes selects incremental inclusive-threshold IF event counts.
const Spikes OutputMode = "spikes"

// Linear selects the cumulative signed final-layer current integral.
const Linear OutputMode = "linear"

// DenseLayer owns row-major output-by-input coefficients and a membrane preload.
type DenseLayer struct {
	inputs, outputs     int
	weights, bias       []float64
	threshold, fraction float64
}

func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

// NewDenseLayer validates and independently copies all coefficients. The numeric
// copy reservation is 16 bytes per coefficient, excluding caller/runtime storage.
func NewDenseLayer(inputs, outputs int, weights, bias []float64, threshold, fraction float64, maxWorkingBytes int) (DenseLayer, error) {
	if inputs <= 0 || outputs <= 0 || maxWorkingBytes <= 0 ||
		new(big.Int).Mul(big.NewInt(int64(inputs)), big.NewInt(int64(outputs))).Cmp(big.NewInt(int64(len(weights)))) != 0 ||
		(bias != nil && len(bias) != outputs) || !finite(threshold) || threshold <= 0 || !finite(fraction) {
		return DenseLayer{}, ErrInvalidInput
	}
	coefficients := new(big.Int).Add(big.NewInt(int64(len(weights))), big.NewInt(int64(len(bias))))
	if coefficients.Mul(coefficients, big.NewInt(16)).Cmp(big.NewInt(int64(maxWorkingBytes))) > 0 {
		return DenseLayer{}, ErrResourceLimit
	}
	for _, v := range weights {
		if !finite(v) {
			return DenseLayer{}, ErrInvalidInput
		}
	}
	for _, v := range bias {
		if !finite(v) {
			return DenseLayer{}, ErrInvalidInput
		}
	}
	owned := append([]float64(nil), weights...)
	var ownedBias []float64
	if bias != nil {
		ownedBias = append([]float64(nil), bias...)
	}
	return DenseLayer{inputs, outputs, owned, ownedBias, threshold, fraction}, nil
}

// ConvertedSNN owns a nonempty connected dense stack and final response mode.
// Coefficients are private; instances can replay concurrently without mutations.
type ConvertedSNN struct {
	layers []DenseLayer
	mode   OutputMode
}

// NewConvertedSNN validates connectivity and owns independent coefficient copies.
func NewConvertedSNN(layers []DenseLayer, mode OutputMode, maxWorkingBytes int) (*ConvertedSNN, error) {
	if len(layers) == 0 || (mode != Spikes && mode != Linear) || maxWorkingBytes <= 0 {
		return nil, ErrInvalidInput
	}
	coefficients := new(big.Int)
	for i, layer := range layers {
		if layer.inputs <= 0 || layer.outputs <= 0 || (i > 0 && layers[i-1].outputs != layer.inputs) {
			return nil, ErrInvalidInput
		}
		coefficients.Add(coefficients, big.NewInt(int64(len(layer.weights))))
		coefficients.Add(coefficients, big.NewInt(int64(len(layer.bias))))
	}
	if coefficients.Mul(coefficients, big.NewInt(16)).Cmp(big.NewInt(int64(maxWorkingBytes))) > 0 {
		return nil, ErrResourceLimit
	}
	owned := make([]DenseLayer, len(layers))
	for i, layer := range layers {
		owned[i] = layer
		owned[i].weights = append([]float64(nil), layer.weights...)
		if layer.bias != nil {
			owned[i].bias = append([]float64(nil), layer.bias...)
		}
	}
	return &ConvertedSNN{owned, mode}, nil
}
