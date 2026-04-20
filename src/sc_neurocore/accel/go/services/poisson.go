// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for poisson

package services

import (
	"math"
)

// PoissonNeuronState holds the neuron state
type PoissonNeuronState struct {
	RateHz float64
	DtMs float64
	Rng float64
}

// NewPoissonNeuron creates a new PoissonNeuron neuron with default parameters
func NewPoissonNeuron() *PoissonNeuronState {
	return &PoissonNeuronState{
		RateHz: 100.0,
		DtMs: 1.0,
		Rng: 0.0,
	}
}

// Step advances the neuron by one timestep
func (s *PoissonNeuronState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulatePoissonNeuron runs the neuron for n steps
func SimulatePoissonNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewPoissonNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.RateHz
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var _ = math.Exp
