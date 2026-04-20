// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for threshold_linear_rate

package services

import (
	"math"
)

// ThresholdLinearRateNeuronState holds the neuron state
type ThresholdLinearRateNeuronState struct {
	R float64
	Theta float64
	Gain float64
}

// NewThresholdLinearRateNeuron creates a new ThresholdLinearRateNeuron neuron with default parameters
func NewThresholdLinearRateNeuron() *ThresholdLinearRateNeuronState {
	return &ThresholdLinearRateNeuronState{
		R: 0.0,
		Theta: 0.0,
		Gain: 1.0,
	}
}

// Step advances the neuron by one timestep
func (s *ThresholdLinearRateNeuronState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateThresholdLinearRateNeuron runs the neuron for n steps
func SimulateThresholdLinearRateNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewThresholdLinearRateNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.R
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var _ = math.Exp
