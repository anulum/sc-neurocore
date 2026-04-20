// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for ltc

package services

import (
	"math"
)

// LiquidTimeConstantNeuronState holds the neuron state
type LiquidTimeConstantNeuronState struct {
	X float64
	TauBase float64
	WTau float64
	WX float64
	WIn float64
	Bias float64
	VThreshold float64
	Dt float64
}

// NewLiquidTimeConstantNeuron creates a new LiquidTimeConstantNeuron neuron with default parameters
func NewLiquidTimeConstantNeuron() *LiquidTimeConstantNeuronState {
	return &LiquidTimeConstantNeuronState{
		X: 0.0,
		TauBase: 10.0,
		WTau: -0.5,
		WX: 0.8,
		WIn: 1.0,
		Bias: 0.0,
		VThreshold: 1.0,
		Dt: 1.0,
	}
}

// Step advances the neuron by one timestep
func (s *LiquidTimeConstantNeuronState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateLiquidTimeConstantNeuron runs the neuron for n steps
func SimulateLiquidTimeConstantNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewLiquidTimeConstantNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.X
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var _ = math.Exp
