// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go borrowed dense coefficient admission

package conversion

import "math/big"

// LayerParameters borrows row-major coefficients until construction finishes.
// Inputs/Outputs are positive widths; Bias is nil or one value per output.
// Threshold is finite and positive; InitialFraction is finite threshold units.
type LayerParameters struct {
	// Inputs is the positive input width.
	Inputs int
	// Outputs is the positive output width.
	Outputs int
	// Weights borrows Outputs*Inputs row-major finite doubles.
	Weights []float64
	// Bias is nil or one finite per-timestep bias per output.
	Bias []float64
	// Threshold is the positive finite inclusive firing threshold.
	Threshold float64
	// InitialFraction is a finite preload measured in threshold units.
	InitialFraction float64
}

// NewConvertedSNNFromParameters admits the complete coefficient reservation
// before independently copying each borrowed layer exactly once. No caller array
// is retained. The numeric reservation is16P bytes, excluding caller/metadata.
// Invalid connectivity/domains return ErrInvalidInput; budget refusal returns
// ErrResourceLimit without a model. Subsequent Replay has its own full reservation.
func NewConvertedSNNFromParameters(source []LayerParameters, mode OutputMode, maximum int) (*ConvertedSNN, error) {
	if len(source) == 0 || (mode != Spikes && mode != Linear) || maximum <= 0 {
		return nil, ErrInvalidInput
	}
	total := new(big.Int)
	for i, layer := range source {
		if layer.Inputs <= 0 || layer.Outputs <= 0 || (i > 0 && source[i-1].Outputs != layer.Inputs) {
			return nil, ErrInvalidInput
		}
		total.Add(total, big.NewInt(int64(len(layer.Weights))))
		total.Add(total, big.NewInt(int64(len(layer.Bias))))
	}
	if total.Mul(total, big.NewInt(16)).Cmp(big.NewInt(int64(maximum))) > 0 {
		return nil, ErrResourceLimit
	}
	owned := make([]DenseLayer, len(source))
	for i, layer := range source {
		value, err := NewDenseLayer(layer.Inputs, layer.Outputs, layer.Weights, layer.Bias, layer.Threshold, layer.InitialFraction, maximum)
		if err != nil {
			return nil, err
		}
		owned[i] = value
	}
	return &ConvertedSNN{owned, mode}, nil
}
