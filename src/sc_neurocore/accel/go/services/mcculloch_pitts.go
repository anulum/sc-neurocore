// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for mcculloch_pitts

package services

import "errors"

var (
	ErrMcCullochPittsInvalidInput = errors.New("mcculloch pitts weighted input must be finite")
	ErrMcCullochPittsInvalidState = errors.New("mcculloch pitts threshold must be finite")
)

// McCullochPittsNeuronState holds the neuron state
type McCullochPittsNeuronState struct {
	Theta float64
}

// NewMcCullochPittsNeuron creates a new McCullochPittsNeuron neuron with default parameters
func NewMcCullochPittsNeuron() *McCullochPittsNeuronState {
	return &McCullochPittsNeuronState{
		Theta: 1.0,
	}
}

// Step advances the stateless threshold unit by one timestep.
func (s *McCullochPittsNeuronState) Step(weightedInput float64) (int, error) {
	if !finite(weightedInput) {
		return 0, ErrMcCullochPittsInvalidInput
	}
	if !ValidateMcCullochPitts(s) {
		return 0, ErrMcCullochPittsInvalidState
	}
	if weightedInput >= s.Theta {
		return 1, nil
	}
	return 0, nil
}

// ValidateMcCullochPitts enforces the finite Heaviside threshold contract.
func ValidateMcCullochPitts(s *McCullochPittsNeuronState) bool {
	return s != nil && finite(s.Theta)
}

// SimulateMcCullochPittsNeuron runs the neuron for n steps
func SimulateMcCullochPittsNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewMcCullochPittsNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result, err := s.Step(iExt)
		if err != nil {
			panic(err)
		}
		trace[t] = s.Theta
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}
